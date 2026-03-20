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
    csv_candidates = [f for f in year_dir.glob("*partidas.csv") if "ids.csv" not in f.name]
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

    col_map = {
        "round": "rodada",
        "home_team": "clube_casa_id",
        "away_team": "clube_visitante_id",
    }
    for src, dst in col_map.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]

    # Hack para score '1 x 0' -> separar em placar_oficial_mandante e placar_oficial_visitante
    if "score" in df.columns and "placar_oficial_mandante" not in df.columns:
        try:
            split_scores = df["score"].str.split(" x ", expand=True)
            if split_scores.shape[1] == 2:
                df["placar_oficial_mandante"] = pd.to_numeric(split_scores[0], errors='coerce').fillna(0)
                df["placar_oficial_visitante"] = pd.to_numeric(split_scores[1], errors='coerce').fillna(0)
        except:
            pass

    for req in ["rodada", "clube_casa_id", "clube_visitante_id", "placar_oficial_mandante", "placar_oficial_visitante"]:
        if req not in df.columns:
            df[req] = 0

    keep_cols = ["rodada", "clube_casa_id", "clube_visitante_id", "placar_oficial_mandante", "placar_oficial_visitante"]
    df = df[keep_cols].copy()
    
    df["rodada"] = pd.to_numeric(df["rodada"], errors='coerce').fillna(0).astype(int)
    df = df[df["rodada"] > 0].copy()
    return df


def load_pontuacoes_year(year_dir: Path) -> pd.DataFrame:
    # 2014-2017: *_scouts_raw.csv; 2018+: rodada-*.csv
    csv_candidates = list(year_dir.glob("*scouts_raw.csv")) + list(year_dir.glob("rodada-*.csv"))
    if not csv_candidates:
        logger.warning(f"Nenhum arquivo de pontuações encontrado em {year_dir}")
        return pd.DataFrame()

    df_list = []
    for f in csv_candidates:
        try:
            df_temp = pd.read_csv(f)
            # Clean up column names dynamically
            df_temp.columns = [c.replace("atletas.", "").replace("atleta.", "") for c in df_temp.columns]
            
            if "rodada_id" in df_temp.columns and "rodada" not in df_temp.columns:
                df_temp["rodada"] = df_temp["rodada_id"]
                
            # If rodada is missing but we're parsing rodada-X.csv
            if "rodada" not in df_temp.columns and f.name.startswith("rodada-"):
                try:
                    df_temp["rodada"] = int(f.stem.split("-")[1])
                except:
                    pass
            
            df_list.append(df_temp)
            logger.info(f"Carregado {f} com {len(df_temp)} registros")
        except Exception as e:
            logger.warning(f"Falha lendo {f}: {e}")

    if not df_list:
        return pd.DataFrame()
    df = pd.concat(df_list, ignore_index=True)

    col_map = {
        "Rodada": "rodada",
        "AtletaID": "atleta_id",
        "Pontos": "pontos",
        "pontos_num": "pontos",
        "ClubeID": "clube_id",
    }
    
    for src, dst in col_map.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]

    for req in ["rodada", "atleta_id", "pontos", "clube_id"]:
        if req not in df.columns:
            df[req] = 0

    keep_cols = ["rodada", "atleta_id", "pontos", "clube_id"]
    df = df[keep_cols].copy()
    
    df["rodada"] = pd.to_numeric(df["rodada"], errors='coerce').fillna(0).astype(int)
    df["atleta_id"] = pd.to_numeric(df["atleta_id"], errors='coerce').fillna(0).astype(int)
    df["pontos"] = pd.to_numeric(df["pontos"], errors='coerce').fillna(0.0).astype(float)
    df["clube_id"] = pd.to_numeric(df["clube_id"], errors='coerce').fillna(0).astype(int)
    
    df = df[df["rodada"] > 0].copy()
    return df


def upsert_dataframe(df: pd.DataFrame, table: str, conn: sqlite3.Connection, if_exists: str = "append"):
    if df.empty:
        return
        
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table})")
    valid_cols = [row[1] for row in cursor.fetchall()]
    
    cols_to_keep = [c for c in df.columns if c in valid_cols]
    df_clean = df[cols_to_keep].copy()
    
    logger.info(f"Gravando {len(df_clean)} linhas em {table} ({if_exists} with IGNORE)")
    
    temp_table = f"temp_{table}"
    df_clean.to_sql(temp_table, conn, if_exists="replace", index=False)
    
    cols_str = ", ".join(cols_to_keep)
    cursor.execute(f"INSERT OR IGNORE INTO {table} ({cols_str}) SELECT {cols_str} FROM {temp_table}")
    cursor.execute(f"DROP TABLE {temp_table}")
    conn.commit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cartola-root", required=True, help="Caminho raiz do repo caRtola")
    ap.add_argument("--db-path", required=True, help="Caminho para o banco SQLite do seu projeto")
    args = ap.parse_args()

    cartola_root = Path(args.cartola_root).expanduser().resolve()
    db_path = Path(args.db_path).expanduser().resolve()

    year_dirs = find_year_dirs(cartola_root)
    logger.info(f"Anos encontrados em caRtola: {[p.name for p in year_dirs]}")

    conn = sqlite3.connect(str(db_path))
    
    # Try to initialize DB scheme just in case it doesn't exist
    try:
        import sys
        # Add project root to sys.path to import src
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from src.data.collector import CartolaDataCollector
        # Create a dummy collector just to init database
        # Requires mocking config for the database path
        import yaml
        dummy_config_path = cartola_root.parent / "temp_config.yaml"
        with open(dummy_config_path, "w") as f:
            yaml.dump({"database": {"path": str(db_path)}}, f)
        collector = CartolaDataCollector(api_client=None, config_path=str(dummy_config_path))
        dummy_config_path.unlink(missing_ok=True)
        logger.info("Banco de dados inicializado/migrado pelo collector.py")
    except Exception as e:
        logger.warning(f"Não foi possível inicializar banco via collector, usando schema existente. Erro: {e}")

    for year_dir in year_dirs:
        logger.info(f"Processando ano {year_dir.name}...")
        part_year = load_partidas_year(year_dir)
        pont_year = load_pontuacoes_year(year_dir)

        if not part_year.empty:
            part_year["ano"] = int(year_dir.name)
            upsert_dataframe(part_year, "partidas", conn, if_exists="append")
            
        if not pont_year.empty:
            pont_year["ano"] = int(year_dir.name)
            upsert_dataframe(pont_year, "pontuacoes", conn, if_exists="append")

    conn.close()
    logger.info("Importação completa.")


if __name__ == "__main__":
    main()
