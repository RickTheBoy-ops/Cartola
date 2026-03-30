import sqlite3
import pandas as pd
from src.ml.optimizer import GeneticTeamOptimizer
from src.optimizer.genetic_strategy import GeneticStrategy
import os
os.environ['PYTHONPATH'] = '.'

def test_genetic():
    conn = sqlite3.connect('data/cartola.db')
    atletas = pd.read_sql_query("SELECT atleta_id, apelido, posicao_id, clube_id FROM atletas", conn)
    
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
    except Exception as e:
        import traceback
        with open('error.log', 'w') as f:
            f.write(traceback.format_exc())

if __name__ == "__main__":
    test_genetic()

if __name__ == "__main__":
    test_genetic()
