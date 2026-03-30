"""Métricas de performance de escalações otimizadas (sem IA).

Este módulo trabalha exclusivamente em cima do SQLite (`data/cartola.db`),
cruzando a tabela `escalacoes` com `pontuacoes` para calcular quanto as
escalações sugeridas realmente pontuaram quando a rodada fechou.

Uso típico:

    from src.analysis.performance import (
        calcular_performance_rodada,
        historico_performance,
    )

    df_rodada = calcular_performance_rodada(db_path, ano=2024, rodada=7)
    df_hist = historico_performance(db_path, ano=2024, ultimas_n_rodadas=10)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd


def _connect(db_path: Path | str) -> sqlite3.Connection:
    return sqlite3.connect(str(db_path))


def calcular_performance_rodada(
    db_path: Path | str,
    ano: int,
    rodada: int,
) -> pd.DataFrame:
    """Calcula desempenho real das escalações de uma rodada.

    Retorna um DataFrame com, para cada combinação de
    (estrategia, formacao, patrimonio):

    - pontos_time: soma dos pontos de todos os atletas da escalação
    - pontos_capitao: pontos do(s) atleta(s) marcado(s) como capitão
    - n_atletas: quantidade de atletas considerados na soma
    """
    conn = _connect(db_path)
    try:
        query = """
            SELECT
                e.ano,
                e.rodada,
                e.estrategia,
                e.formacao,
                e.patrimonio,
                COUNT(e.atleta_id)            AS n_atletas,
                SUM(p.pontos)                 AS pontos_time,
                SUM(CASE WHEN e.eh_capitao = 1 THEN p.pontos END)
                                               AS pontos_capitao
            FROM escalacoes e
            JOIN pontuacoes p
              ON e.atleta_id = p.atleta_id
             AND e.rodada    = p.rodada
             AND e.ano       = p.ano
            WHERE e.ano = ? AND e.rodada = ?
            GROUP BY
                e.ano,
                e.rodada,
                e.estrategia,
                e.formacao,
                e.patrimonio
            ORDER BY pontos_time DESC
        """
        df = pd.read_sql_query(query, conn, params=(ano, rodada))
    finally:
        conn.close()
    return df


def historico_performance(
    db_path: Path | str,
    ano: int,
    ultimas_n_rodadas: Optional[int] = None,
) -> pd.DataFrame:
    """Retorna histórico de performance agregada das escalações.

    - Se `ultimas_n_rodadas` for informado, limita às últimas N rodadas
      com dados em `escalacoes`.
    - Agrupa por (estrategia, formacao, patrimonio) somando pontos ao
      longo do tempo e calculando média por rodada.
    """
    conn = _connect(db_path)
    try:
        base_query = """
            SELECT DISTINCT ano, rodada
            FROM escalacoes
            WHERE ano = ?
            ORDER BY rodada DESC
        """
        rodadas_df = pd.read_sql_query(base_query, conn, params=(ano,))
        if rodadas_df.empty:
            return pd.DataFrame()

        if ultimas_n_rodadas is not None:
            rodadas_sel = rodadas_df.tail(ultimas_n_rodadas)["rodada"].tolist()
        else:
            rodadas_sel = rodadas_df["rodada"].tolist()

        placeholder = ",".join("?" for _ in rodadas_sel)
        query = f"""
            SELECT
                e.ano,
                e.rodada,
                e.estrategia,
                e.formacao,
                e.patrimonio,
                COUNT(e.atleta_id)            AS n_atletas,
                SUM(p.pontos)                 AS pontos_time,
                SUM(CASE WHEN e.eh_capitao = 1 THEN p.pontos END)
                                               AS pontos_capitao
            FROM escalacoes e
            JOIN pontuacoes p
              ON e.atleta_id = p.atleta_id
             AND e.rodada    = p.rodada
             AND e.ano       = p.ano
            WHERE e.ano = ? AND e.rodada IN ({placeholder})
            GROUP BY
                e.ano,
                e.rodada,
                e.estrategia,
                e.formacao,
                e.patrimonio
        """

        params = (ano, *rodadas_sel)
        df = pd.read_sql_query(query, conn, params=params)

        if df.empty:
            return df

        # Agregação por estratégia/formação ao longo do tempo
        agg = (
            df.groupby(["ano", "estrategia", "formacao", "patrimonio"], as_index=False)
              .agg(
                  rodadas=(