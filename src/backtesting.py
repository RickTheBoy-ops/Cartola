"""
Módulo de Backtesting Automático para Cartola FC.
Simula escalações recomendadas pelo sistema de IA em rodadas históricas
e calcula métricas de performance vs. realidade.
"""

from __future__ import annotations

import os
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
RESULTS_PATH = DATA_DIR / "backtesting_results.csv"


# ---------------------------------------------------------------------------
# Carregamento de dados históricos
# ---------------------------------------------------------------------------

def carregar_dados_historicos() -> pd.DataFrame:
    """Carrega dados históricos de rodadas do diretório data/.

    Procura por arquivos CSV ou JSON com colunas:
    rodada, atleta_id, nome, posicao, clube, pontos_reais, preco, media.

    Returns:
        DataFrame consolidado com histórico de todas as rodadas.
    """
    dfs: list[pd.DataFrame] = []

    for arquivo in sorted(DATA_DIR.glob("rodada_*.csv")):
        try:
            df = pd.read_csv(arquivo)
            if "rodada" not in df.columns:
                rodada_num = int(re.search(r"(\d+)", arquivo.stem).group(1)) if re.search(r"(\d+)", arquivo.stem) else 0
                df["rodada"] = rodada_num
            dfs.append(df)
        except Exception as exc:
            logger.warning("Erro ao carregar %s: %s", arquivo, exc)

    for arquivo in sorted(DATA_DIR.glob("rodada_*.json")):
        try:
            with arquivo.open() as f:
                dados = json.load(f)
            df = pd.DataFrame(dados if isinstance(dados, list) else dados.get("atletas", []))
            if "rodada" not in df.columns:
                import re
                rodada_num = int(re.search(r"(\d+)", arquivo.stem).group(1)) if re.search(r"(\d+)", arquivo.stem) else 0
                df["rodada"] = rodada_num
            dfs.append(df)
        except Exception as exc:
            logger.warning("Erro ao carregar %s: %s", arquivo, exc)

    if not dfs:
        logger.warning("Nenhum arquivo de rodada encontrado em %s. Usando dados sintéticos.", DATA_DIR)
        return _gerar_dados_sinteticos()

    return pd.concat(dfs, ignore_index=True)


def _gerar_dados_sinteticos(n_rodadas: int = 10, n_jogadores: int = 50) -> pd.DataFrame:
    """Gera dados sintéticos para demonstração quando não há histórico real."""
    import re
    rng = np.random.default_rng(42)
    registros = []
    posicoes = ["GOL", "ZAG", "LAT", "MEI", "ATA", "TEC"]
    nomes = [f"Jogador_{i}" for i in range(1, n_jogadores + 1)]

    for rodada in range(1, n_rodadas + 1):
        for i, nome in enumerate(nomes):
            pontos = float(rng.normal(loc=5.5, scale=3.0))
            pontos = max(0.0, pontos)
            registros.append({
                "rodada": rodada,
                "atleta_id": i + 1,
                "nome": nome,
                "posicao": posicoes[i % len(posicoes)],
                "clube": f"Clube_{(i % 10) + 1}",
                "pontos_reais": round(pontos, 2),
                "preco": round(rng.uniform(4.0, 25.0), 2),
                "media": round(rng.uniform(3.0, 9.0), 2),
            })

    return pd.DataFrame(registros)


# ---------------------------------------------------------------------------
# Simulação de escalação por rodada
# ---------------------------------------------------------------------------

def simular_escalacao_rodada(
    df_rodada: pd.DataFrame,
    budget: float = 100.0,
    formacao: str = "4-3-3",
) -> pd.DataFrame:
    """Simula a escalação otimizada para uma rodada usando média histórica.

    Usa estratégia gulosa: ordena por média/preço dentro de cada posição.

    Args:
        df_rodada: DataFrame com jogadores disponíveis na rodada.
        budget: Orçamento disponível em cartoletas.
        formacao: Formação tática desejada.

    Returns:
        DataFrame com os 11 titulares selecionados + técnico.
    """
    FORMACOES: dict[str, dict[str, int]] = {
        "4-3-3": {"GOL": 1, "ZAG": 2, "LAT": 2, "MEI": 3, "ATA": 3, "TEC": 1},
        "4-4-2": {"GOL": 1, "ZAG": 2, "LAT": 2, "MEI": 4, "ATA": 2, "TEC": 1},
        "3-5-2": {"GOL": 1, "ZAG": 3, "LAT": 0, "MEI": 5, "ATA": 2, "TEC": 1},
        "4-5-1": {"GOL": 1, "ZAG": 2, "LAT": 2, "MEI": 5, "ATA": 1, "TEC": 1},
    }

    slots = FORMACOES.get(formacao, FORMACOES["4-3-3"])
    df_work = df_rodada.copy()
    df_work["valor"] = df_work["media"] / df_work["preco"].replace(0, 1)
    df_work = df_work.sort_values("valor", ascending=False)

    selecionados: list[pd.Series] = []
    gasto = 0.0

    for posicao, quantidade in slots.items():
        candidatos = df_work[df_work["posicao"] == posicao].head(quantidade * 3)
        count = 0
        for _, row in candidatos.iterrows():
            if count >= quantidade:
                break
            if gasto + row["preco"] <= budget:
                selecionados.append(row)
                gasto += row["preco"]
                count += 1

    return pd.DataFrame(selecionados)


def melhor_escalacao_possivel(
    df_rodada: pd.DataFrame,
    budget: float = 100.0,
    formacao: str = "4-3-3",
) -> pd.DataFrame:
    """Retorna a melhor escalação possível (hindsight) usando pontos reais."""
    FORMACOES: dict[str, dict[str, int]] = {
        "4-3-3": {"GOL": 1, "ZAG": 2, "LAT": 2, "MEI": 3, "ATA": 3, "TEC": 1},
        "4-4-2": {"GOL": 1, "ZAG": 2, "LAT": 2, "MEI": 4, "ATA": 2, "TEC": 1},
        "3-5-2": {"GOL": 1, "ZAG": 3, "LAT": 0, "MEI": 5, "ATA": 2, "TEC": 1},
        "4-5-1": {"GOL": 1, "ZAG": 2, "LAT": 2, "MEI": 5, "ATA": 1, "TEC": 1},
    }

    slots = FORMACOES.get(formacao, FORMACOES["4-3-3"])
    df_work = df_rodada.sort_values("pontos_reais", ascending=False)
    selecionados: list[pd.Series] = []
    gasto = 0.0

    for posicao, quantidade in slots.items():
        candidatos = df_work[df_work["posicao"] == posicao]
        count = 0
        for _, row in candidatos.iterrows():
            if count >= quantidade:
                break
            if gasto + row["preco"] <= budget:
                selecionados.append(row)
                gasto += row["preco"]
                count += 1

    return pd.DataFrame(selecionados)


# ---------------------------------------------------------------------------
# Engine principal de backtesting
# ---------------------------------------------------------------------------

def executar_backtesting(
    budget: float = 100.0,
    formacao: str = "4-3-3",
    salvar_csv: bool = True,
) -> pd.DataFrame:
    """Executa o backtesting completo em todas as rodadas disponíveis.

    Args:
        budget: Orçamento em cartoletas.
        formacao: Formação tática.
        salvar_csv: Se True, salva resultados em data/backtesting_results.csv.

    Returns:
        DataFrame com métricas por rodada.
    """
    df_historico = carregar_dados_historicos()
    rodadas = sorted(df_historico["rodada"].unique())

    relatorio_rodadas: list[dict] = []

    for rodada in rodadas:
        df_rodada = df_historico[df_historico["rodada"] == rodada].copy()

        if len(df_rodada) < 11:
            logger.warning("Rodada %s com poucos jogadores, pulando.", rodada)
            continue

        # Escalação simulada pela IA (usa médias históricas anteriores)
        df_escalacao_ia = simular_escalacao_rodada(df_rodada, budget, formacao)
        # Melhor escalação possível (hindsight)
        df_melhor = melhor_escalacao_possivel(df_rodada, budget, formacao)

        pontos_ia = df_escalacao_ia["pontos_reais"].sum() if len(df_escalacao_ia) > 0 else 0.0
        pontos_melhor = df_melhor["pontos_reais"].sum() if len(df_melhor) > 0 else 0.0
        pontos_media_geral = df_rodada["pontos_reais"].mean() * 11

        # Calcula MAE e RMSE entre pontos esperados (média) e reais
        if len(df_escalacao_ia) > 0:
            esperado = df_escalacao_ia["media"].values
            real = df_escalacao_ia["pontos_reais"].values
            mae = float(np.mean(np.abs(esperado - real)))
            rmse = float(np.sqrt(np.mean((esperado - real) ** 2)))
        else:
            mae, rmse = 0.0, 0.0

        relatorio_rodadas.append({
            "rodada": int(rodada),
            "pontos_ia": round(pontos_ia, 2),
            "pontos_melhor_possivel": round(pontos_melhor, 2),
            "pontos_media_geral": round(pontos_media_geral, 2),
            "delta_vs_melhor": round(pontos_melhor - pontos_ia, 2),
            "delta_vs_media": round(pontos_ia - pontos_media_geral, 2),
            "superou_media": pontos_ia > pontos_media_geral,
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "n_jogadores_ia": len(df_escalacao_ia),
            "time_ia": df_escalacao_ia["nome"].tolist() if len(df_escalacao_ia) > 0 else [],
        })

    df_resultado = pd.DataFrame(relatorio_rodadas)

    if salvar_csv and not df_resultado.empty:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df_resultado.to_csv(RESULTS_PATH, index=False)
        logger.info("Resultados salvos em %s", RESULTS_PATH)

    return df_resultado


def calcular_metricas_gerais(df_resultado: pd.DataFrame) -> dict:
    """Calcula métricas consolidadas do backtesting.

    Args:
        df_resultado: DataFrame retornado por executar_backtesting().

    Returns:
        Dicionário com métricas agregadas.
    """
    if df_resultado.empty:
        return {}

    return {
        "total_rodadas": len(df_resultado),
        "media_pontos_ia": round(df_resultado["pontos_ia"].mean(), 2),
        "media_pontos_melhor": round(df_resultado["pontos_melhor_possivel"].mean(), 2),
        "media_pontos_geral": round(df_resultado["pontos_media_geral"].mean(), 2),
        "pct_superou_media": round(df_resultado["superou_media"].mean() * 100, 1),
        "mae_medio": round(df_resultado["mae"].mean(), 2),
        "rmse_medio": round(df_resultado["rmse"].mean(), 2),
        "melhor_rodada": int(df_resultado.loc[df_resultado["pontos_ia"].idxmax(), "rodada"]),
        "pior_rodada": int(df_resultado.loc[df_resultado["pontos_ia"].idxmin(), "rodada"]),
    }


if __name__ == "__main__":
    import json
    df = executar_backtesting(budget=100.0, formacao="4-3-3")
    metricas = calcular_metricas_gerais(df)
    print("\n=== MÉTRICAS GERAIS DO BACKTESTING ===")
    print(json.dumps(metricas, indent=2, ensure_ascii=False))
    print("\n=== RESULTADO POR RODADA ===")
    print(df[["rodada", "pontos_ia", "pontos_melhor_possivel", "delta_vs_melhor", "superou_media", "mae"]].to_string(index=False))
