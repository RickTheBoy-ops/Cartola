import sys
from pathlib import Path
sys.path.insert(0, str(Path("c:/RASPAGEM CARTOLA/Cartola").resolve()))

from scripts.import_cartola_history import load_partidas_year, load_pontuacoes_year, upsert_dataframe
import sqlite3
import pandas as pd

cartola_root = Path("c:/RASPAGEM CARTOLA/caRtola_repo")
year_dir = cartola_root / "data" / "01_raw" / "2019"

db_path = Path("c:/RASPAGEM CARTOLA/Cartola/data/cartola_2019_test.db")
if db_path.exists():
    db_path.unlink()

# Init schema
from src.data.collector import CartolaDataCollector
import yaml
dummy_config = cartola_root / "temp_config.yaml"
with open(dummy_config, "w") as f:
    yaml.dump({"database": {"path": str(db_path)}}, f)
CartolaDataCollector(None, str(dummy_config))

conn = sqlite3.connect(db_path)
part = load_partidas_year(year_dir)
if not part.empty:
    part["ano"] = 2019
    upsert_dataframe(part, "partidas", conn, if_exists="append")

pont = load_pontuacoes_year(year_dir)
if not pont.empty:
    pont["ano"] = 2019
    upsert_dataframe(pont, "pontuacoes", conn, if_exists="append")

print("---")
print("PARTIDAS (amostra de 5 linhas):")
print(pd.read_sql("SELECT * FROM partidas LIMIT 5", conn).to_string(index=False))

print("\n---")
print("PONTUACOES (amostra de 10 linhas):")
print(pd.read_sql("SELECT * FROM pontuacoes LIMIT 10", conn).to_string(index=False))

conn.close()
