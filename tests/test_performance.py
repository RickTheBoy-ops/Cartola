import sqlite3
from pathlib import Path

import pandas as pd

from src.analysis.performance import calcular_performance_rodada, historico_performance


def _build_fake_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "cartola_test.db"
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            CREATE TABLE pontuacoes (
                ano INTEGER NOT NULL,
                rodada INTEGER NOT NULL,
                atleta_id INTEGER NOT NULL,
                pontos REAL NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE escalacoes (
                ano INTEGER NOT NULL,
                rodada INTEGER NOT NULL,
                estrategia TEXT NOT NULL,
                formacao TEXT NOT NULL,
                patrimonio REAL NOT NULL,
                atleta_id INTEGER NOT NULL,
                posicao_id INTEGER NOT NULL,
                eh_capitao INTEGER DEFAULT 0
            )
            """
        )

        # Inserir dados mínimos para duas rodadas com mesma estratégia
        data_pont = [
            (2026, 1, 10, 8.0),
            (2026, 1, 11, 5.5),
            (2026, 2, 10, 7.0),
            (2026, 2, 11, 9.0),
        ]
        conn.executemany(
            "INSERT INTO pontuacoes (ano, rodada, atleta_id, pontos) VALUES (?, ?, ?, ?)",
            data_pont,
        )

        data_esc = [
            (2026, 1, "mega", "4-3-3", 100.0, 10, 5, 1),
            (2026, 1, "mega", "4-3-3", 100.0, 11, 4, 0),
            (2026, 2, "mega", "4-3-3", 100.0, 10, 5, 0),
            (2026, 2, "mega", "4-3-3", 100.0, 11, 4, 1),
        ]
        conn.executemany(
            "INSERT INTO escalacoes (ano, rodada, estrategia, formacao, patrimonio, atleta_id, posicao_id, eh_capitao) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            data_esc,
        )
        conn.commit()
    finally:
        conn.close()

    return db_path


def test_calcular_performance_rodada(tmp_path: Path) -> None:
    db_path = _build_fake_db(tmp_path)

    df = calcular_performance_rodada(db_path, ano=2026, rodada=1)

    assert not df.empty
    assert df.iloc[0]["pontos_time"] == 13.5
    assert df.iloc[0]["pontos_capitao"] == 8.0


def test_historico_performance(tmp_path: Path) -> None:
    db_path = _build_fake_db(tmp_path)

    df_hist = historico_performance(db_path, ano=2026, ultimas_n_rodadas=2)

    assert not df_hist.empty
    # Duas rodadas com 2 atletas cada
    assert df_hist.iloc[0]["rodadas"] == 2
    # Total de pontos somados nas duas rodadas
    assert df_hist.iloc[0]["pontos_totais"] == 29.5
    # Média por rodada
    assert df_hist.iloc[0]["media_por_rodada"] == 29.5 / 2
