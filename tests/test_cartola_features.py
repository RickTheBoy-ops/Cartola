import pytest
import pandas as pd
import numpy as np
from src.ml.features import FeatureEngineer


@pytest.fixture
def mock_hist_and_partidas():
    df_hist = pd.DataFrame({
        'atleta_id': [1, 1, 1, 2, 2, 2],
        'clube_id': [10, 10, 10, 20, 20, 20],
        'rodada': [1, 2, 3, 1, 2, 3],
        'pontos': [2.0, 5.0, 8.0, -1.0, 3.0, 6.0],
        'preco': [10.0, 10.5, 11.2, 5.0, 4.8, 5.5],
        'posicao_id': [4, 4, 4, 1, 1, 1],
    })
    df_partidas = pd.DataFrame({
        'rodada': [1, 2, 3],
        'clube_casa_id': [10, 20, 10],
        'clube_visitante_id': [20, 10, 20],
        'placar_oficial_mandante': [2, 1, 3],
        'placar_oficial_visitante': [0, 1, 1],
        'aproveitamento_mandante': [0.6, 0.5, 0.7],
        'aproveitamento_visitante': [0.4, 0.5, 0.3],
    })
    return df_hist, df_partidas


def test_engineer_base_features(mock_hist_and_partidas):
    df_hist, df_partidas = mock_hist_and_partidas
    df_feat = FeatureEngineer.engineer_all_features(df_hist, df_partidas)
    assert 'pontos_media_3' in df_feat.columns
    assert 'mando_casa' in df_feat.columns
    assert 'pontos_ewm_3' in df_feat.columns
    assert 'atleta_id' in df_feat.columns
    assert 'rodada' in df_feat.columns


def test_engineer_media_sliding(mock_hist_and_partidas):
    df_hist, df_partidas = mock_hist_and_partidas
    df_feat = FeatureEngineer.engineer_all_features(df_hist, df_partidas)
    a1_r3 = df_feat[(df_feat['atleta_id'] == 1) & (df_feat['rodada'] == 3)].iloc[0]
    assert a1_r3['pontos_media_3'] > 0
    assert not pd.isna(a1_r3['pontos_media_3'])


# -----------------------------------------------------------------------
# Testes: add_score_no_cleansheets (R: score.no.cleansheets)
# -----------------------------------------------------------------------

def test_score_sem_sg_via_pontos_num():
    """Caminho 1: pontos_num + SG disponíveis → subtrai SG * 5.0."""
    df = pd.DataFrame({"pontos_num": [13.0], "SG": [1]})
    result = FeatureEngineer.add_score_no_cleansheets(df)
    assert "score_sem_sg" in result.columns
    # 13 - 1*5 = 8
    assert result["score_sem_sg"].iloc[0] == pytest.approx(8.0, abs=1e-6)


def test_score_sem_sg_via_pontos():
    """Caminho 1 alternativo: usa coluna 'pontos' quando pontos_num ausente."""
    df = pd.DataFrame({"pontos": [13.0], "SG": [1]})
    result = FeatureEngineer.add_score_no_cleansheets(df)
    assert result["score_sem_sg"].iloc[0] == pytest.approx(8.0, abs=1e-6)


def test_score_sem_sg_via_scouts_fallback():
    """Caminho 2: sem coluna pontos, reconstrói via scouts sem bônus SG."""
    df = pd.DataFrame({
        "atleta_id": [1],
        "G": [1], "SG": [1],
        "CA": [0], "FC": [0], "GC": [0], "CV": [0],
        "FS": [0], "A": [0], "FT": [0], "FD": [0],
        "FF": [0], "I": [0], "PP": [0], "DS": [0],
        "GS": [0], "DE": [0], "DP": [0], "PE": [0],
    })
    result = FeatureEngineer.add_score_no_cleansheets(df)
    assert "score_sem_sg" in result.columns
    # G=1 × 8.0, SG excluído → score_sem_sg == 8.0
    assert result["score_sem_sg"].iloc[0] == pytest.approx(8.0, abs=1e-6)


def test_score_sem_sg_sg_zero_nao_altera():
    """Jogador sem SG: score_sem_sg == pontos bruto."""
    df = pd.DataFrame({"pontos": [10.5], "SG": [0]})
    result = FeatureEngineer.add_score_no_cleansheets(df)
    assert result["score_sem_sg"].iloc[0] == pytest.approx(10.5, abs=1e-6)


def test_score_sem_sg_sem_scouts_retorna_zero():
    """Sem scouts nem pontos, não deve quebrar: retorna 0.0."""
    df = pd.DataFrame({"atleta_id": [1, 2]})
    result = FeatureEngineer.add_score_no_cleansheets(df)
    assert "score_sem_sg" in result.columns
    assert (result["score_sem_sg"] == 0.0).all()
