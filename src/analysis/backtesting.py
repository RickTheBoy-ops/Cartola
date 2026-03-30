#!/usr/bin/env python3
"""
src/analysis/backtesting.py

Motor de Backtesting Automático — Cartola FC Optimizer

Simula ganhos e pontuação das escalações otimizadas em rodadas anteriores,
comparando a estratégia do modelo contra a realidade histórica.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────
POSICAO_NOME = {1: "GOL", 2: "LAT", 3: "ZAG", 4: "MEI", 5: "ATA", 6: "TEC"}
FORMACOES_SLOTS = {
    "4-3-3": {1: 1, 2: 2, 3: 2, 4: 3, 5: 3, 6: 1},
    "4-4-2": {1: 1, 2: 2, 3: 2, 4: 4, 5: 2, 6: 1},
    "3-5-2": {1: 1, 2: 2, 3: 1, 4: 5, 5: 2, 6: 1},
    "3-4-3": {1: 1, 2: 2, 3: 1, 4: 4, 5: 3, 6: 1},
    "4-5-1": {1: 1, 2: 2, 3: 2, 4: 5, 5: 1, 6: 1},
    "5-3-2": {1: 1, 2: 2, 3: 3, 4: 3, 5: 2, 6: 1},
    "5-4-1": {1: 1, 2: 2, 3: 3, 4: 4, 5: 1, 6: 1},
}


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _load_pontuacoes(db_path: Path, ano: int) -> pd.DataFrame:
    """Carrega toda a tabela de pontuações do SQLite para o ano informado."""
    conn = sqlite3.connect(str(db_path))
    df = pd.read_sql_query(
        """
        SELECT p.atleta_id, p.rodada, p.pontos, p.preco, p.media,
               a.posicao_id, a.clube_id, a.apelido
        FROM pontuacoes p
        JOIN atletas a ON p.atleta_id = a.atleta_id
        WHERE p.ano = ?
        ORDER BY p.atleta_id, p.rodada
        """,
        conn,
        params=(ano,),
    )
    conn.close()
    return df


def _load_escalacoes(db_path: Path, ano: int) -> pd.DataFrame:
    """Carrega escalações salvas pelo pipeline (tabela 'escalacoes')."""
    conn = sqlite3.connect(str(db_path))
    try:
        df = pd.read_sql_query(
            "SELECT * FROM escalacoes WHERE ano = ? ORDER BY rodada DESC",
            conn,
            params=(ano,),
        )
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df


def _escalacao_heuristica(
    pontuacoes: pd.DataFrame,
    rodada: int,
    patrimonio: float,
    formacao: str,
    janela: int = 5,
) -> pd.DataFrame:
    """
    Simula a escalação que o modelo teria feito na `rodada` usando apenas
    o histórico anterior (sem look-ahead).  Estratégia: média ponderada EWM
    das últimas `janela` rodadas — escolhe os N melhores por posição
    respeitando o orçamento.
    """
    slots = FORMACOES_SLOTS.get(formacao, FORMACOES_SLOTS["4-3-3"])
    historico = pontuacoes[pontuacoes["rodada"] < rodada].copy()

    if historico.empty:
        return pd.DataFrame()

    # Calcula EWM por atleta nas últimas `janela` aparições
    ultimas = (
        historico.sort_values("rodada")
        .groupby("atleta_id")
        .tail(janela)
        .copy()
    )
    stats = (
        ultimas.groupby("atleta_id")
        .apply(
            lambda g: pd.Series(
                {
                    "pontos_ewm": g["pontos"].ewm(span=3, min_periods=1).mean().iloc[-1],
                    "preco_atual": g.sort_values("rodada")["preco"].iloc[-1],
                    "posicao_id": g["posicao_id"].iloc[-1],
                    "apelido": g["apelido"].iloc[-1],
                    "clube_id": g["clube_id"].iloc[-1],
                }
            )
        )
        .reset_index()
    )

    # Filtrar atletas com preco > 0 (disponíveis no mercado)
    stats = stats[stats["preco_atual"] > 0]

    team_rows = []
    orcamento = patrimonio

    for pos_id, n_vagas in slots.items():
        candidatos = (
            stats[stats["posicao_id"] == pos_id]
            .sort_values("pontos_ewm", ascending=False)
        )
        selecionados = 0
        for _, row in candidatos.iterrows():
            if selecionados >= n_vagas:
                break
            if row["preco_atual"] <= orcamento:
                team_rows.append(row)
                orcamento -= row["preco_atual"]
                selecionados += 1

    if not team_rows:
        return pd.DataFrame()

    return pd.DataFrame(team_rows)


def _pontos_reais_time(
    time_df: pd.DataFrame,
    pontuacoes: pd.DataFrame,
    rodada: int,
) -> dict:
    """Cruza os atletas escalados com a pontuação real daquela rodada."""
    if time_df.empty:
        return {"pontos": 0.0, "n_atletas": 0, "cobertura": 0.0}

    reais = pontuacoes[
        (pontuacoes["rodada"] == rodada)
        & (pontuacoes["atleta_id"].isin(time_df["atleta_id"]))
    ]

    pontos = float(reais["pontos"].sum())
    cobertura = len(reais) / len(time_df) if len(time_df) > 0 else 0.0
    return {"pontos": pontos, "n_atletas": len(reais), "cobertura": cobertura}


# ─────────────────────────────────────────────────────────────
# FUNÇÃO PRINCIPAL
# ─────────────────────────────────────────────────────────────

def run_backtesting(
    db_path: Path,
    ano: int,
    formacao: str = "4-3-3",
    patrimonio: float = 100.0,
    n_rodadas: int = 10,
    janela_ewm: int = 5,
    usar_escalacoes_salvas: bool = True,
) -> pd.DataFrame:
    """
    Executa o backtesting nas últimas `n_rodadas` disponíveis no banco.

    Parâmetros
    ----------
    db_path : Path
        Caminho para o SQLite do Cartola.
    ano : int
        Temporada (ex: 2024).
    formacao : str
        Formação tática simulada (ex: '4-3-3').
    patrimonio : float
        Patrimônio (Cartoletas) disponível por rodada.
    n_rodadas : int
        Quantas rodadas passadas simular.
    janela_ewm : int
        Janela de rodadas anteriores usada para calcular média EWM.
    usar_escalacoes_salvas : bool
        Se True, usa as escalações reais salvas pelo pipeline quando
        disponíveis; caso contrário, re-simula via heurística EWM.

    Retorna
    -------
    pd.DataFrame com colunas:
        rodada, pontos_reais, pontos_media_mercado,
        patrimonio_gasto, cobertura_pct, ganho_vs_mercado,
        estrategia, formacao
    """
    pontuacoes = _load_pontuacoes(db_path, ano)
    if pontuacoes.empty:
        return pd.DataFrame()

    rodadas_disponiveis = sorted(pontuacoes["rodada"].unique())
    # Precisa de ao menos 2 rodadas (1 pra estimar + 1 pra avaliar)
    if len(rodadas_disponiveis) < 2:
        return pd.DataFrame()

    rodadas_para_testar = rodadas_disponiveis[-n_rodadas:]

    # Tenta usar escalações salvas para identificar estratégia e formação reais
    escalacoes_salvas = pd.DataFrame()
    if usar_escalacoes_salvas:
        escalacoes_salvas = _load_escalacoes(db_path, ano)

    results = []
    for rodada in rodadas_para_testar:
        # ── Média do mercado nessa rodada ─────────────────
        todos_rodada = pontuacoes[pontuacoes["rodada"] == rodada]
        media_mercado = float(todos_rodada["pontos"].mean()) if not todos_rodada.empty else 0.0

        # ── Verificar escalação salva ─────────────────────
        esc_salva = pd.DataFrame()
        estrategia_usada = "heuristica_ewm"
        formacao_usada = formacao

        if not escalacoes_salvas.empty and "rodada" in escalacoes_salvas.columns:
            esc_rodada = escalacoes_salvas[escalacoes_salvas["rodada"] == rodada]
            if not esc_rodada.empty:
                row_esc = esc_rodada.iloc[0]
                estrategia_usada = row_esc.get("estrategia", "salva")
                formacao_usada = row_esc.get("formacao", formacao)

                # Tentar reconstruir time a partir dos atleta_ids salvos
                atletas_ids_col = (
                    "atletas_ids" if "atletas_ids" in row_esc.index else None
                )
                if atletas_ids_col and isinstance(row_esc[atletas_ids_col], str):
                    try:
                        import json
                        ids_list = json.loads(row_esc[atletas_ids_col])
                        esc_salva = pontuacoes[
                            (pontuacoes["rodada"] == rodada)
                            & (pontuacoes["atleta_id"].isin(ids_list))
                        ][["atleta_id", "preco", "pontos"]]
                    except Exception:
                        esc_salva = pd.DataFrame()

        # ── Fallback: simular heurística ──────────────────
        if esc_salva.empty:
            time_simulado = _escalacao_heuristica(
                pontuacoes, rodada, patrimonio, formacao, janela=janela_ewm
            )
        else:
            time_simulado = esc_salva.rename(columns={"pontos": "pontos_real"}).copy()
            time_simulado["preco_atual"] = time_simulado.get("preco", pd.Series([0] * len(time_simulado)))

        # ── Pontos reais do time simulado ─────────────────
        pts_info = _pontos_reais_time(time_simulado, pontuacoes, rodada)

        patrimonio_gasto = float(time_simulado["preco_atual"].sum()) if "preco_atual" in time_simulado.columns else 0.0
        n_atletas_time = len(time_simulado)
        media_time = float(n_atletas_time * media_mercado) if n_atletas_time > 0 else 0.0

        results.append(
            {
                "rodada": rodada,
                "pontos_reais": round(pts_info["pontos"], 2),
                "pontos_media_mercado": round(media_mercado, 2),
                "pontos_medio_time_mercado": round(media_time, 2),
                "patrimonio_gasto": round(patrimonio_gasto, 2),
                "n_atletas": pts_info["n_atletas"],
                "cobertura_pct": round(pts_info["cobertura"] * 100, 1),
                "ganho_vs_mercado": round(pts_info["pontos"] - media_time, 2),
                "estrategia": estrategia_usada,
                "formacao": formacao_usada,
            }
        )

    df_result = pd.DataFrame(results)
    return df_result


# ─────────────────────────────────────────────────────────────
# MÉTRICAS RESUMIDAS
# ─────────────────────────────────────────────────────────────

def summarize_backtesting(bt_df: pd.DataFrame) -> dict:
    """Retorna métricas agregadas do backtest para exibição no dashboard."""
    if bt_df.empty:
        return {}

    rodadas_positivas = (bt_df["ganho_vs_mercado"] > 0).sum()
    total_rodadas = len(bt_df)
    win_rate = (rodadas_positivas / total_rodadas * 100) if total_rodadas > 0 else 0.0

    return {
        "total_rodadas": total_rodadas,
        "media_pontos": round(float(bt_df["pontos_reais"].mean()), 2),
        "max_pontos": round(float(bt_df["pontos_reais"].max()), 2),
        "min_pontos": round(float(bt_df["pontos_reais"].min()), 2),
        "ganho_total_vs_mercado": round(float(bt_df["ganho_vs_mercado"].sum()), 2),
        "win_rate_pct": round(win_rate, 1),
        "rodadas_positivas": int(rodadas_positivas),
        "media_cobertura": round(float(bt_df["cobertura_pct"].mean()), 1),
    }
