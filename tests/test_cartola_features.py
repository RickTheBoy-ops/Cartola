import pytest
import pandas as pd
import numpy as np
from src.ml.features import FeatureEngineer

@pytest.fixture
def mock_hist_and_partidas():
    # Historico basico
    df_hist = pd.DataFrame({
        'atleta_id': [1, 1, 1, 2, 2, 2],
        'clube_id': [10, 10, 10, 20, 20, 20],
        'rodada': [1, 2, 3, 1, 2, 3],
        'pontos': [2.0, 5.0, 8.0, -1.0, 3.0, 6.0],
        'preco': [10.0, 10.5, 11.2, 5.0, 4.8, 5.5],
        'posicao_id': [4, 4, 4, 1, 1, 1]
    })
    
    # Partidas basicas (Clube 10 joga em casa, Clube 20 fora)
    df_partidas = pd.DataFrame({
        'rodada': [1, 2, 3, 1, 2, 3],
        'clube_casa_id': [10, 30, 10, 40, 20, 40],
        'clube_visitante_id': [20, 10, 40, 20, 30, 20],
        'placar_oficial_mandante': [2, 1, 3, 0, 1, 1],
        'placar_oficial_visitante': [0, 1, 1, 0, 1, 2]
    })
    
    return df_hist, df_partidas

def test_engineer_base_features(mock_hist_and_partidas):
    df_hist, df_partidas = mock_hist_and_partidas
    
    # Engineer features
    df_feat = FeatureEngineer.engineer_all_features(df_hist, df_partidas)
    
    # Assert
    assert 'media_ultimas_3' in df_feat.columns
    assert 'mando_casa' in df_feat.columns
    assert 'pontos_ewm' in df_feat.columns
    
    # O dataframe resultante nao pode perder as colunas base pre-criadas
    assert 'atleta_id' in df_feat.columns
    assert 'rodada' in df_feat.columns

def test_engineer_media_sliding(mock_hist_and_partidas):
    df_hist, df_partidas = mock_hist_and_partidas
    df_feat = FeatureEngineer.engineer_all_features(df_hist, df_partidas)
    
    # Atleta 1 rodada 3
    a1_r3 = df_feat[(df_feat['atleta_id'] == 1) & (df_feat['rodada'] == 3)].iloc[0]
    
    # A media_ultimas_3 usa dados passados (Shift), entao deve conter a media das rodadas anteriores.
    assert a1_r3['media_ultimas_3'] > 0
    assert not pd.isna(a1_r3['media_ultimas_3'])
