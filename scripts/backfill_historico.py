#!/usr/bin/env python3
"""Backfill de histórico de pontuações no SQLite.

Este script usa o endpoint autenticado `get_pontuacao_atleta` para
preencher rodadas antigas na tabela `pontuacoes` para atletas já
cadastrados na tabela `atletas`.

Limitações:
- Depende do formato retornado pela API do Cartola FC no endpoint de
  pontuação do atleta. O mapeamento de campos pode precisar de ajustes
  conforme o payload real.
- Respeita a UNIQUE(ano, rodada, atleta_id), usando INSERT OR IGNORE
  para não sobrescrever dados já existentes.

Uso sugerido (a partir da raiz do repo):

    python scripts/backfill_historico.py --min-rodada 1 --ano 2024

"""

import argparse
import logging
import sqlite3
from pathlib import Path
from typing import Iterable, Optional

from dotenv import load_dotenv

from src.api.client import CartolaAPIClient, CartolaAPIError
from src.data.collector import CartolaDataCollector

ROOT_DIR = Path(__file__).resolve().parents[1]


def _get_existing_rounds(conn: sqlite3.Connection, atleta_id: int) -> set[int]:
    cur = conn.execute(
        "SELECT DISTINCT rodada FROM pontuacoes WHERE atleta_id = ?", (atleta_id,)
    )
    return {r[0] for r in cur.fetchall()}


def backfill_atleta(
    client: CartolaAPIClient,
    conn: sqlite3.Connection,
    atleta_id: int,
    ano_padrao: int,
    min_rodada: int,
) -> int:
    """Baixa histórico de um atleta e insere rodadas ausentes.

    Retorna o número de registros inseridos.
    """
    try:
        data = client.get_pontuacao_atleta(atleta_id)
    except CartolaAPIError as e:
        logging.warning(
            "Erro na API ao buscar histórico do atleta %s: %s", atleta_id, e
        )
        return 0

    # Tentativa genérica de extrair a lista de registros de pontuação.
    # Em alguns formatos a API pode retornar diretamente uma lista, em
    # outros um dict com uma chave contendo a lista.
    registros = None
    if isinstance(data, list):
        registros = data
    elif isinstance(data, dict):
        # Heurística: procurar por alguma chave que pareça conter a
        # lista de pontuações.
        for key in ("pontos", "historico", "pontuacoes", "rodadas"):
            val = data.get(key)
            if isinstance(val, list):
                registros = val
                break
        if registros is None:
            # Se não encontrar, tentar tratar o próprio dict como um
            # único registro (caso raro).
            registros = [data]
    else:
        logging.warning(
            "Formato inesperado para histórico do atleta %s: %r",
            atleta_id,
            type(data),
        )
        return 0

    existentes = _get_existing_rounds(conn, atleta_id)
    inseridos = 0

    for item in registros:
        if not isinstance(item, dict):
            continue
        rodada = item.get("rodada")
        if rodada is None:
            continue
        if rodada < min_rodada:
            continue
        if rodada in existentes:
            continue

        conn.execute(
            """
            INSERT OR IGNORE INTO pontuacoes
                (atleta_id, rodada, ano, pontos, preco, variacao, media,
                 jogos, minutos_jogados)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                atleta_id,
                rodada,
                item.get("ano", ano_padrao),
                item.get("pontos", 0.0),
                item.get("preco", 0.0),
                item.get("variacao", 0.0),
                item.get("media", 0.0),
                item.get("jogos", 0),
                item.get("minutos_jogados", 0),
            ),
        )
        inseridos += 1

    return inseridos


def iter_atletas(conn: sqlite3.Connection) -> Iterable[int]:
    cur = conn.execute("SELECT atleta_id FROM atletas ORDER BY atleta_id")
    for (atleta_id,) in cur.fetchall():
        yield int(atleta_id)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill de histórico de pontuações no SQLite para atletas "
            "já cadastrados. Requer autenticação na API Cartola FC via .env."
        )
    )
    parser.add_argument(
        "--min-rodada",
        type=int,
        default=1,
        help="Primeira rodada a considerar (rodadas menores serão ignoradas)",
    )
    parser.add_argument(
        "--ano",
        type=int,
        default=2024,
        help="Ano padrão a ser gravado na coluna 'ano' (padrão: 2024)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Carrega variáveis de ambiente, incluindo credenciais Cartola/GLB_TOKEN
    load_dotenv(ROOT_DIR / ".env")

    client = CartolaAPIClient(config_path=str(ROOT_DIR / "config.yaml"))
    collector = CartolaDataCollector(
        api_client=client, config_path=str(ROOT_DIR / "config.yaml")
    )

    conn = sqlite3.connect(str(collector.db_path))

    total_inseridos = 0
    for atleta_id in iter_atletas(conn):
        inseridos = backfill_atleta(
            client,
            conn,
            atleta_id=atleta_id,
            ano_padrao=args.ano,
            min_rodada=args.min_rodada,
        )
        if inseridos:
            logging.info(
                "Atleta %s: %d rodadas inseridas (>= %d)",
                atleta_id,
                inseridos,
                args.min_rodada,
            )
            total_inseridos += inseridos

    conn.commit()
    conn.close()

    logging.info("Backfill concluído. Registros inseridos: %d", total_inseridos)


if __name__ == "__main__":
    main()
