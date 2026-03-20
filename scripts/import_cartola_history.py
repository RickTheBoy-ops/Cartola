#!/usr/bin/env python3
"""
Importa histórico de dados do repositório caRtola
para o banco SQLite usado pelo projeto Cartola.

Uso:
    python scripts/import_cartola_history.py \
        --cartola-root /caminho/para/caRtola \
        --db-path data/cartola.db
"""

import argparse
import logging
import os
from pathlib import Path
from typing import List

import pandas as pd
import sqlite3

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def find_year_dirs(cartola_root: Path) -> List[Path]:
    data_root = cartola_root / "data" / "01_raw"
    if not data_root.exists():
        raise FileNotFoundError(f"Pasta não encontrada: {data_root}")
    year_dirs = [p for p in data_root.iterdir() if p.is_dir() and p.name.isdigit()]
    return sorted(year_dirs, key=lambda p: int(p.name))


def load_partidas_year(year_dir: Path) -> pd.DataFrame:
    """
    Adapte este loader conforme o esquema real do caRtola.
    Aqui assumo um CSV 'partidas.csv' com colunas mínimas:
      rodada, clube_casa_id, clube_visitante_id,
      placar_oficial_mandante, placar_oficial_visitante, data
    """
    csv_candidates = [f for f in year_dir.glob("**/*partidas*.csv")]
    if not csv_candidates:
        logger.warning(f"Nenhum arquivo de partidas encontrado em {year_dir}")
        return pd.DataFrame()

    df_list = []
    for f in csv_candidates:
        try:
            df_list.append(pd.read_csv(f))
            logger.info(f"Carregado {f}")
        except Exception as e:
            logger.warning(f"Falha lendo {f}: {e}")

    if not df_list:
        return pd.DataFrame()
    df = pd.concat(df_list, ignore_index=True)

    # Normalizar campos básicos (ajuste conforme nomes reais)
    col_map = {
        "rodada": "rodada",
        "clube_casa_id": "clube_casa_id",
        "clube_visitante_id": "clube_visitante_id",
        "placar_mandante": "placar_oficial_mandante",
        "placar_visitante": "placar_oficial_visitante",
    }
    for src, dst in col_map.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]

    keep_cols = [
        "rodada",
        "clube_casa_id",
        "clube_visitante_id",
        "placar_oficial_mandante",
        "placar_oficial_visitante",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()
    df["rodada"] = df["rodada"].astype(int)
    return df


def load_pontuacoes_year(year_dir: Path) -> pd.DataFrame:
    """
    Adapte este loader conforme o esquema real do caRtola.
    Assumimos um CSV 'pontuacoes.csv' com colunas:
      rodada, atleta_id, pontos, clube_id
    """
    csv_candidates = [f for f in year_dir.glob("**/*pontuac*.csv")]
    if not csv_candidates:
        logger.warning(f"Nenhum arquivo de pontuações encontrado em {year_dir}")
        return pd.DataFrame()

    df_list = []
    for f in csv_candidates:
        try:
            df_list.append(pd.read_csv(f))
            logger.info(f"Carregado {f}")
        except Exception as e:
            logger.warning(f"Falha lendo {f}: {e}")

    if not df_list:
        return pd.DataFrame()
    df = pd.concat(df_list, ignore_index=True)

    col_map = {
        "rodada": "rodada",
        "atleta_id": "atleta_id",
        "pontos_num": "pontos",
        "pontos": "pontos",
        "clube_id": "clube_id",
    }
    for src, dst in col_map.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]

    keep_cols = ["rodada", "atleta_id", "pontos", "clube_id"]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()
    df["rodada"] = df["rodada"].astype(int)
    df["atleta_id"] = df["atleta_id"].astype(int)
    return df


def upsert_dataframe(df: pd.DataFrame, table: str, conn: sqlite3.Connection, if_exists: str = "append"):
    if df.empty:
        return
    logger.info(f"Gravando {len(df)} linhas em {table} ({if_exists})")
    df.to_sql(table, conn, if_exists=if_exists, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cartola-root", required=True, help="Caminho raiz do repo caRtola")
    ap.add_argument("--db-path", required=True, help="Caminho para o banco SQLite do seu projeto")
    args = ap.parse_args()

    cartola_root = Path(args.cartola_root).expanduser().resolve()
    db_path = Path(args.db_path).expanduser().resolve()

    year_dirs = find_year_dirs(cartola_root)
    logger.info(f"Anos encontrados em caRtola: {[p.name for p in year_dirs]}")

    all_partidas = []
    all_pontuacoes = []

    for year_dir in year_dirs:
        logger.info(f"Processando ano {year_dir.name}...")
        part_year = load_partidas_year(year_dir)
        pont_year = load_pontuacoes_year(year_dir)

        if not part_year.empty:
            part_year["ano"] = int(year_dir.name)
            all_partidas.append(part_year)
        if not pont_year.empty:
            pont_year["ano"] = int(year_dir.name)
            all_pontuacoes.append(pont_year)

    if not all_partidas and not all_pontuacoes:
        logger.error("Nenhum dado válido encontrado. Verifique o layout dos CSVs.")
        return

    conn = sqlite3.connect(str(db_path))

    if all_partidas:
        df_partidas = pd.concat(all_partidas, ignore_index=True)
        # Ajuste: se sua tabela 'partidas' tiver colunas extras, você pode preencher depois
        upsert_dataframe(df_partidas, "partidas", conn, if_exists="append")

    if all_pontuacoes:
        df_pontuacoes = pd.concat(all_pontuacoes, ignore_index=True)
        upsert_dataframe(df_pontuacoes, "pontuacoes", conn, if_exists="append")

    conn.close()
    logger.info("Importação completa.")


if __name__ == "__main__":
    main()
