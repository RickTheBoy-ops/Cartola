"""Configurações e fixtures para testes pytest."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def sample_atletas_df():
    """Fixture com DataFrame de exemplo de atletas."""
    return pd.DataFrame({
        'atleta_id': [1, 2, 3, 4, 5],
        'apelido': ['Neymar', 'Messi', 'CR7', 'Mbappé', 'Haaland'],
        'clube_id': [1, 2, 1, 3, 2],
        'posicao_id': [5, 5, 5, 5, 4],  # Atacantes e meia
        'preco': [25.5, 30.0, 28.5, 22.0, 26.5],
        'pontos_rodada': [10.5, 12.0, 8.5, 15.0, 11.0],
        'media_pontos': [8.5, 10.2, 9.1, 11.5, 9.8],
        'jogos': [5, 5, 5, 5, 5],
        'status_id': [7, 7, 7, 7, 7]  # Provável
    })


@pytest.fixture
def sample_partidas_df():
    """Fixture com DataFrame de exemplo de partidas."""
    return pd.DataFrame({
        'partida_id': [1, 2, 3],
        'clube_casa_id': [1, 2, 3],
        'clube_visitante_id': [2, 3, 1],
        'rodada': [1, 1, 1],
        'placar_casa': [2, 1, 0],
        'placar_visitante': [1, 1, 2]
    })


@pytest.fixture
def sample_historico_df():
    """Fixture com DataFrame de histórico de pontuações."""
    rodadas = []
    for rodada in range(1, 6):
        for atleta_id in range(1, 6):
            rodadas.append({
                'atleta_id': atleta_id,
                'rodada': rodada,
                'pontos': np.random.uniform(0, 20),
                'clube_id': (atleta_id % 3) + 1,
                'posicao_id': 5 if atleta_id <= 3 else 4
            })
    return pd.DataFrame(rodadas)


@pytest.fixture
def sample_mercado_status():
    """Fixture com status de mercado de exemplo."""
    return {
        'rodada_atual': 10,
        'status_mercado': 1,  # Aberto
        'fechamento': '2026-03-15T14:00:00',
        'abertura': '2026-03-14T10:00:00'
    }


@pytest.fixture
def mock_api_client(monkeypatch):
    """Mock do cliente de API."""
    class MockAPIClient:
        def __init__(self, *args, **kwargs):
            pass
        
        def get_mercado_status(self):
            return {'rodada_atual': 10, 'status_mercado': 1}
        
        def get_atletas(self):
            return {'atletas': {}}
    
    return MockAPIClient


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Diretório temporário para dados de teste."""
    return tmp_path_factory.mktemp("test_data")


@pytest.fixture(autouse=True)
def reset_cache():
    """Limpa cache antes de cada teste."""
    from src.utils.cache import clear_cache
    clear_cache()
    yield
    clear_cache()
