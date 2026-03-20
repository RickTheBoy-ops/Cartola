"""
PoissonTeamRanking
==================
Dois GLMs Poisson separados — um para gols do mandante, um para gols do visitante.
Equivalente ao `rank.teams(..., family="poisson")` do pacote R `fbRanks`.

Referência: caRtola/src/R/team_data_create_features.R

  E[gols_mandante] = exp(µ + Σ(dummies_home) + Σ(dummies_away))
  E[gols_visitante] = modelo simétrico com os mesmos regressores

  P(SG do mandante) = P(gols_visitante = 0) = exp(-λ_away)
"""
import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_statsmodels_ok = False
try:
    import statsmodels.api as sm
    _statsmodels_ok = True
except ImportError:
    logger.warning(
        "statsmodels não encontrado — PoissonTeamRanking indisponível. "
        "Instale com: pip install statsmodels>=0.14.0"
    )


class PoissonTeamRanking:
    """
    Modelo Poisson simples por time mandante/visitante.

    Objetivo principal: estimar λ de gols esperados por time, de forma a
    derivar P(SG) ≈ exp(-λ_gols_sofridos).
    """

    MIN_PARTIDAS = 50  # mínimo razoável para convergência do GLM

    def __init__(self) -> None:
        self.model_home_: Optional[object] = None
        self.model_away_: Optional[object] = None
        self.home_cols_: Optional[List[str]] = None
        self.away_cols_: Optional[List[str]] = None

    # ------------------------------------------------------------------
    def _build_design_matrices(
        self, partidas_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Constrói matrizes de design para dois modelos:
          - gols_mandante ~ dummies(home_id) + dummies(away_id)
          - gols_visitante ~ dummies(home_id) + dummies(away_id)

        Espera colunas:
          clube_casa_id, clube_visitante_id,
          placar_oficial_mandante, placar_oficial_visitante
        """
        required = [
            "clube_casa_id", "clube_visitante_id",
            "placar_oficial_mandante", "placar_oficial_visitante",
        ]
        missing = [c for c in required if c not in partidas_df.columns]
        if missing:
            raise ValueError(f"PoissonTeamRanking: colunas ausentes: {missing}")

        df = (
            partidas_df[required]
            .dropna(subset=["clube_casa_id", "clube_visitante_id"])
            .copy()
        )
        df["clube_casa_id"] = df["clube_casa_id"].astype(int)
        df["clube_visitante_id"] = df["clube_visitante_id"].astype(int)

        home_dummies = pd.get_dummies(df["clube_casa_id"], prefix="home")
        away_dummies = pd.get_dummies(df["clube_visitante_id"], prefix="away")
        # pandas >= 1.5 returns bool dtype for dummies; statsmodels needs float
        X_base = home_dummies.join(away_dummies).astype(float)

        # Modelo para gols do mandante
        y_home = (
            pd.to_numeric(df["placar_oficial_mandante"], errors="coerce")
            .fillna(0).astype(int)
        )
        X_home = sm.add_constant(X_base.copy(), has_constant="add")

        # Modelo para gols do visitante (mesma estrutura)
        y_away = (
            pd.to_numeric(df["placar_oficial_visitante"], errors="coerce")
            .fillna(0).astype(int)
        )
        X_away = sm.add_constant(X_base.copy(), has_constant="add")

        return X_home, y_home, X_away, y_away

    # ------------------------------------------------------------------
    def fit(self, partidas_df: pd.DataFrame) -> "PoissonTeamRanking":
        """
        Ajusta dois GLMs Poisson (gols mandante + gols visitante).
        Propaga exceção ao caller para decidir fallback.
        """
        if not _statsmodels_ok:
            raise RuntimeError("statsmodels não instalado.")
        if partidas_df is None or partidas_df.empty:
            raise ValueError("partidas_df vazio.")

        X_home, y_home, X_away, y_away = self._build_design_matrices(partidas_df)

        if len(X_home) < self.MIN_PARTIDAS:
            raise ValueError(
                f"Poucas partidas ({len(X_home)}) — mínimo {self.MIN_PARTIDAS}."
            )

        logger.info("PoissonTeamRanking: ajustando modelo (mandante)...")
        self.model_home_ = sm.GLM(
            y_home, X_home, family=sm.families.Poisson()
        ).fit(disp=False)

        logger.info("PoissonTeamRanking: ajustando modelo (visitante)...")
        self.model_away_ = sm.GLM(
            y_away, X_away, family=sm.families.Poisson()
        ).fit(disp=False)

        self.home_cols_ = list(X_home.columns)
        self.away_cols_ = list(X_away.columns)

        logger.info(
            "✅ PoissonTeamRanking: modelos ajustados (%d partidas).", len(X_home)
        )
        return self

    # ------------------------------------------------------------------
    def _build_exog_row(self, home_id: int, away_id: int, cols: List[str]) -> pd.DataFrame:
        """Constrói vetor de features para um par (home, away)."""
        exog = pd.DataFrame([[0.0] * len(cols)], columns=cols)
        for key, val in [(f"home_{home_id}", 1.0), (f"away_{away_id}", 1.0)]:
            if key in exog.columns:
                exog.at[0, key] = val
        if "const" in exog.columns:
            exog.at[0, "const"] = 1.0
        return exog

    # ------------------------------------------------------------------
    def predict_goals(self, home_id: int, away_id: int) -> Tuple[float, float]:
        """
        Retorna (λ_home, λ_away) = gols esperados de mandante e visitante.

        Nota: model.predict() já retorna λ (não o preditor linear),
        portanto NÃO aplicamos exp() novamente.
        """
        if self.model_home_ is None or self.model_away_ is None:
            raise RuntimeError("Chame fit() antes de predict_goals().")

        exog_h = self._build_exog_row(home_id, away_id, self.home_cols_)
        exog_a = self._build_exog_row(home_id, away_id, self.away_cols_)

        lam_home = float(self.model_home_.predict(exog_h).iloc[0])
        lam_away = float(self.model_away_.predict(exog_a).iloc[0])
        return lam_home, lam_away

    # ------------------------------------------------------------------
    def predict_clean_sheet_prob(self, home_id: int, away_id: int) -> float:
        """
        P(SG do mandante) ≈ P(gols_visitante = 0) = exp(-λ_away).

        Retorna float em [0, 1].
        """
        try:
            _, lam_away = self.predict_goals(home_id, away_id)
            return float(np.clip(np.exp(-lam_away), 0.0, 1.0))
        except Exception as exc:
            logger.warning("predict_clean_sheet_prob falhou: %s", exc)
            return 0.30  # fallback conservador

    # ------------------------------------------------------------------
    def get_rankings(self) -> pd.DataFrame:
        """
        Ranking relativo por time: attack (λ_home), defense (1/(1+λ_away)), strength.
        """
        if self.model_home_ is None or self.home_cols_ is None:
            raise RuntimeError("Chame fit() antes de get_rankings().")

        home_ids = sorted(
            int(c.split("_", 1)[1])
            for c in self.home_cols_
            if c.startswith("home_") and c[5:].isdigit()
        )
        away_ids = sorted(
            int(c.split("_", 1)[1])
            for c in (self.away_cols_ or [])
            if c.startswith("away_") and c[5:].isdigit()
        )
        team_ids = sorted(set(home_ids) | set(away_ids))

        rows = []
        for tid in team_ids:
            try:
                lam_h, lam_a = self.predict_goals(home_id=tid, away_id=tid)
                rows.append({
                    "team_id": tid,
                    "attack":   round(lam_h, 4),
                    "defense":  round(1.0 / (1.0 + lam_a), 4),
                    "strength": round(lam_h - lam_a, 4),
                })
            except Exception:
                pass

        return pd.DataFrame(rows).sort_values("strength", ascending=False).reset_index(drop=True)
