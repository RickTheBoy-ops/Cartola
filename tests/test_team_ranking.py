"""
Tests for PoissonTeamRanking (src/models/team_ranking.py).

Uses a synthetic dataset of 60 matches — enough to ensure GLM convergence
for a small N of teams without requiring real data.
"""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels", reason="statsmodels not installed")

from src.models.team_ranking import PoissonTeamRanking


@pytest.fixture
def synthetic_partidas():
    """60 synthetic matches between 4 clubs (IDs 10, 20, 30, 40)."""
    rng = np.random.default_rng(42)
    teams = [10, 20, 30, 40]
    rows = []
    for _ in range(60):
        h, a = rng.choice(teams, size=2, replace=False)
        gh = int(rng.poisson(1.5))
        ga = int(rng.poisson(1.0))
        rows.append({
            "clube_casa_id": int(h),
            "clube_visitante_id": int(a),
            "placar_oficial_mandante": gh,
            "placar_oficial_visitante": ga,
        })
    return pd.DataFrame(rows)


def test_fit_runs_without_error(synthetic_partidas):
    """fit() must complete without raising."""
    model = PoissonTeamRanking()
    model.fit(synthetic_partidas)
    assert model.model_home_ is not None
    assert model.model_away_ is not None


def test_predict_clean_sheet_prob_range(synthetic_partidas):
    """predict_clean_sheet_prob() must return a value in [0, 1]."""
    model = PoissonTeamRanking().fit(synthetic_partidas)
    p = model.predict_clean_sheet_prob(home_id=10, away_id=20)
    assert isinstance(p, float)
    assert 0.0 <= p <= 1.0


def test_predict_goals_positive(synthetic_partidas):
    """predict_goals() must return positive lambdas."""
    model = PoissonTeamRanking().fit(synthetic_partidas)
    lam_h, lam_a = model.predict_goals(home_id=10, away_id=20)
    assert lam_h > 0.0
    assert lam_a > 0.0


def test_get_rankings_structure(synthetic_partidas):
    """get_rankings() must return a DataFrame with required columns."""
    model = PoissonTeamRanking().fit(synthetic_partidas)
    ranks = model.get_rankings()
    assert isinstance(ranks, pd.DataFrame)
    for col in ("team_id", "attack", "defense", "strength"):
        assert col in ranks.columns, f"Missing column: {col}"
    assert len(ranks) > 0


def test_fit_raises_on_too_few_matches():
    """fit() must raise ValueError when fewer than MIN_PARTIDAS rows are given."""
    tiny = pd.DataFrame({
        "clube_casa_id": [10, 20],
        "clube_visitante_id": [20, 10],
        "placar_oficial_mandante": [1, 0],
        "placar_oficial_visitante": [0, 1],
    })
    model = PoissonTeamRanking()
    with pytest.raises((ValueError, Exception)):
        model.fit(tiny)


def test_predict_without_fit_raises():
    """Calling predict_goals before fit must raise RuntimeError."""
    model = PoissonTeamRanking()
    with pytest.raises(RuntimeError):
        model.predict_goals(home_id=10, away_id=20)
