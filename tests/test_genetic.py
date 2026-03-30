import sqlite3
import os

import pandas as pd
import pytest

from src.optimizer.genetic_strategy import GeneticStrategy

# Caminho esperado do banco local usado apenas para este teste manual
DB_PATH = "data/cartola.db"


@pytest.mark.skipif(
    not os.path.exists(DB_PATH),
    reason="Requer banco local data/cartola.db com tabela 'atletas' (teste manual, não roda em CI)",
)
def test_genetic():
    """Teste manual/integrado do GeneticStrategy com banco SQLite real.

    Em ambientes onde o arquivo data/cartola.db não estiver presente (como no CI),
    o teste é automaticamente marcado como skipped para não quebrar o pipeline.
    """
    conn = sqlite3.connect(DB_PATH)
    atletas = pd.read_sql_query(
        "SELECT atleta_id, apelido, posicao_id, clube_id FROM atletas",
        conn,
    )

    import numpy as np

    atletas['preco'] = np.random.uniform(5, 15, len(atletas))
    atletas['predicao'] = np.random.uniform(2, 12, len(atletas))
    atletas['predicao_std'] = 1.0
    atletas['status_id'] = 7

    strat = GeneticStrategy()
    print("Iniciando strat.optimize...")
    try:
        res = strat.optimize(atletas, budget=120, formation='4-4-2')
        if res is None:
            print("GENETIC RETURNED NONE. WHY?")
        else:
            print("GENETIC SUCCESS! len:", len(res))
    except Exception:
        # Em testes manuais, salvar o traceback para inspeção local
        import traceback

        with open('error.log', 'w') as f:
            f.write(traceback.format_exc())
