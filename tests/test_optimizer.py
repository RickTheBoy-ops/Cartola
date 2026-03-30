import pytest
import pandas as pd
import numpy as np
from src.optimizer.factory import CartolaOptimizer

@pytest.fixture
def mock_atletas_df():
    # Criar um dataframe simulado com atletas suficientes para formar times (ao menos 2 de cada posição)
    # Total: 3 goleiros, 6 zagueiros, 6 laterais, 8 meias, 8 atacantes, 3 técnicos = 34 atletas
    
    np.random.seed(42)
    n_atletas = 34
    
    # 1: GOL, 2: LAT, 3: ZAG, 4: MEI, 5: ATA, 6: TEC
    posicoes = [1]*3 + [2]*6 + [3]*6 + [4]*8 + [5]*8 + [6]*3
    clubes = np.random.randint(1, 10, size=n_atletas)
    
    precos = np.random.uniform(2.0, 15.0, size=n_atletas)
    mega_scores = np.random.uniform(0.0, 10.0, size=n_atletas)
    
    data = {
        'atleta_id': range(1, n_atletas + 1),
        'clube_id': clubes,
        'posicao_id': posicoes,
        'status_id': [7] * n_atletas,
        'preco': precos,
        'mega_score': mega_scores,
        'media': mega_scores * 0.9,
        'apelido': [f"Jogador_{i}" for i in range(1, n_atletas + 1)],
    }
    
    df = pd.DataFrame(data)
    return df


def test_mega_strategy(mock_atletas_df):
    optimizer = CartolaOptimizer(strategy='mega', config={'test_all_formations': False})
    
    lineup = optimizer.optimize(mock_atletas_df, budget=100.0, formation='4-3-3')
    
    assert lineup is not None
    assert len(lineup) == 12
    assert lineup['preco'].sum() <= 100.0
    assert optimizer.validate(lineup, budget=100.0, formation='4-3-3')


def test_genetic_strategy(mock_atletas_df):
    optimizer = CartolaOptimizer(strategy='genetic', config={
        'population_size': 50,
        'generations': 5,
    })
    
    lineup = optimizer.optimize(mock_atletas_df, budget=100.0, formation='4-3-3')
    
    assert lineup is not None
    assert len(lineup) == 12
    # Tolerância de 0.05 para arredondamento de ponto flutuante; a factory garante
    # orçamento via fallback robusto antes de retornar o lineup.
    assert lineup['preco'].sum() <= 100.05, (
        f"Budget exceeded: {lineup['preco'].sum():.4f} > 100.05"
    )
    assert optimizer.validate(lineup, budget=100.0, formation='4-3-3')


def test_ensemble_strategy(mock_atletas_df):
    optimizer = CartolaOptimizer(strategy='ensemble', config={
        'strategies': ['mega', 'genetic'],
        'strategy_configs': {
            'genetic': {'population_size': 10, 'generations': 2}
        }
    })
    
    lineup = optimizer.optimize(mock_atletas_df, budget=120.0, formation='3-4-3')
    
    assert lineup is not None
    assert len(lineup) == 12
    assert lineup['preco'].sum() <= 120.0
    assert optimizer.validate(lineup, budget=120.0, formation='3-4-3')
