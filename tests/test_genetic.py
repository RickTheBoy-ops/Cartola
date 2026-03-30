import numpy as np
import pandas as pd

from src.optimizer.genetic_strategy import GeneticStrategy


def test_genetic():
    """Teste do GeneticStrategy usando dados mockados em memória.

    Este teste não depende de nenhum banco SQLite local e pode rodar
    normalmente no GitHub Actions.
    """
    np.random.seed(42)
    n_atletas = 30

    # 1: GOL, 2: LAT, 3: ZAG, 4: MEI, 5: ATA, 6: TEC
    posicoes = [1] * 3 + [2] * 6 + [3] * 6 + [4] * 8 + [5] * 5 + [6] * 2

    atletas = pd.DataFrame({
        'atleta_id': range(1, n_atletas + 1),
        'apelido': [f"Jogador_{i}" for i in range(1, n_atletas + 1)],
        'posicao_id': posicoes,
        'clube_id': np.random.randint(1, 10, n_atletas),
        'preco': np.random.uniform(5, 15, n_atletas),
        'predicao': np.random.uniform(2, 12, n_atletas),
        'predicao_std': [1.0] * n_atletas,
        'status_id': [7] * n_atletas,
    })

    strat = GeneticStrategy()

    lineup = strat.optimize(atletas, budget=120.0, formation='4-4-2')

    assert lineup is not None
    assert len(lineup) == 12
    assert lineup['preco'].sum() <= 120.0
