import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TeamMatchupRow:
    rodada: int
    clube_id: int
    adversario_id: int
    mando_casa: int
    attack_strength_recent: float
    defense_strength_recent: float
    win_hist_prob: float
    clean_sheet_hist_prob: float
    prob_gol_odds: float
    prob_sg_odds: float
    attack_multiplier: float
    defense_multiplier: float
    win_probability: float
    clean_sheet_probability: float
    btts_no_probability: float


class MatchupAnalyzer:
    """
    Motor central de análise de confrontos por time/rodada.

    Constrói multiplicadores táticos de ataque/defesa, probabilidade de SG
    e probabilidade de vitória por clube, para serem aplicados nos atletas.
    """

    def __init__(
        self,
        rodada: int,
        partidas_df: pd.DataFrame,
        historico_partidas_df: pd.DataFrame,
        odds_por_clube: Optional[pd.DataFrame] = None,
        janela_recente: int = 5,
    ):
        self.rodada = rodada
        self.partidas_df = partidas_df.copy()
        self.historico = historico_partidas_df.copy()
        self.janela = janela_recente

        for col in [
            "clube_casa_id",
            "clube_visitante_id",
            "placar_oficial_mandante",
            "placar_oficial_visitante",
        ]:
            if col not in self.partidas_df.columns:
                self.partidas_df[col] = np.nan

        self.odds_por_clube = odds_por_clube

    def _build_resultados_por_clube(self) -> pd.DataFrame:
        """Expande partidas em uma linha por clube por jogo (mandante/visitante)."""
        base = self.historico[self.historico["rodada"] < self.rodada].copy()

        mandante = base[
            ["rodada", "clube_casa_id", "placar_oficial_mandante", "placar_oficial_visitante"]
        ].rename(
            columns={
                "clube_casa_id": "clube_id",
                "placar_oficial_mandante": "gols_pro",
                "placar_oficial_visitante": "gols_contra",
            }
        )
        mandante["mando_casa"] = 1

        visitante = base[
            ["rodada", "clube_visitante_id", "placar_oficial_visitante", "placar_oficial_mandante"]
        ].rename(
            columns={
                "clube_visitante_id": "clube_id",
                "placar_oficial_visitante": "gols_pro",
                "placar_oficial_mandante": "gols_contra",
            }
        )
        visitante["mando_casa"] = 0

        resultados = pd.concat([mandante, visitante], ignore_index=True)
        resultados = resultados.dropna(subset=["clube_id"])
        resultados["clube_id"] = resultados["clube_id"].astype(int)
        return resultados

    def _calc_form_metrics(self, resultados: pd.DataFrame) -> pd.DataFrame:
        """Calcula ataque/defesa recente + prob histórica de SG e vitória."""
        resultados = resultados.sort_values(["clube_id", "rodada"])

        def agg_clube(group: pd.DataFrame) -> pd.Series:
            recent = group.tail(self.janela)

            atk_home = recent.loc[recent["mando_casa"] == 1, "gols_pro"].mean()
            atk_away = recent.loc[recent["mando_casa"] == 0, "gols_pro"].mean()
            def_home = 1.0 / (1.0 + recent.loc[recent["mando_casa"] == 1, "gols_contra"].mean())
            def_away = 1.0 / (1.0 + recent.loc[recent["mando_casa"] == 0, "gols_contra"].mean())

            win_hist = (recent["gols_pro"] > recent["gols_contra"]).mean()
            cs_hist = (recent["gols_contra"] == 0).mean()

            return pd.Series(
                {
                    "attack_home": float(np.nan_to_num(atk_home, nan=0.0)),
                    "attack_away": float(np.nan_to_num(atk_away, nan=0.0)),
                    "defense_home": float(np.nan_to_num(def_home, nan=0.5)),
                    "defense_away": float(np.nan_to_num(def_away, nan=0.5)),
                    "win_hist_prob": float(np.nan_to_num(win_hist, nan=0.5)),
                    "cs_hist_prob": float(np.nan_to_num(cs_hist, nan=0.3)),
                }
            )

        metrics = resultados.groupby("clube_id").apply(agg_clube).reset_index()
        return metrics

    def build_team_matchups(self) -> pd.DataFrame:
        """Gera uma linha por clube na rodada atual com todos os multiplicadores."""
        resultados = self._build_resultados_por_clube()
        if resultados.empty:
            logger.warning("MatchupAnalyzer: sem histórico suficiente de partidas.")
            return pd.DataFrame()

        metrics = self._calc_form_metrics(resultados)

        atk_ref = (metrics["attack_home"] + metrics["attack_away"]).replace(0, np.nan).mean()
        def_ref = (metrics["defense_home"] + metrics["defense_away"]).replace(0, np.nan).mean()
        atk_ref = atk_ref if not np.isnan(atk_ref) else 1.0
        def_ref = def_ref if not np.isnan(def_ref) else 1.0

        odds_df = None
        if self.odds_por_clube is not None:
            if isinstance(self.odds_por_clube, pd.DataFrame):
                odds_df = self.odds_por_clube.copy()
            else:
                odds_df = (
                    pd.DataFrame.from_dict(self.odds_por_clube, orient="index")
                    .reset_index()
                    .rename(columns={"index": "clube_id"})
                )

        jogos = self.partidas_df[self.partidas_df["rodada"] == self.rodada].copy()
        if jogos.empty:
            logger.warning("MatchupAnalyzer: nenhuma partida encontrada para a rodada %s", self.rodada)
            return pd.DataFrame()

        rows = []
        for _, p in jogos.iterrows():
            ca = int(p["clube_casa_id"])
            cb = int(p["clube_visitante_id"])

            for clube_id, adversario_id, mando in [(ca, cb, 1), (cb, ca, 0)]:
                m = metrics[metrics["clube_id"] == clube_id]
                if m.empty:
                    atk_recent = 1.0
                    def_recent = 1.0
                    win_hist = 0.5
                    cs_hist = 0.3
                else:
                    row = m.iloc[0]
                    if mando == 1:
                        atk_recent = row["attack_home"]
                        def_recent = row["defense_home"]
                    else:
                        atk_recent = row["attack_away"]
                        def_recent = row["defense_away"]
                    win_hist = row["win_hist_prob"]
                    cs_hist = row["cs_hist_prob"]

                atk_mult = 0.9 + 0.4 * (atk_recent / atk_ref)
                def_mult = 0.9 + 0.4 * (def_recent / def_ref)

                prob_gol_odds = 0.55
                prob_sg_odds = 0.30
                if odds_df is not None and "clube_id" in odds_df.columns:
                    o = odds_df[odds_df["clube_id"] == clube_id]
                    if not o.empty:
                        prob_gol_odds = float(o.iloc[0].get("prob_gol", prob_gol_odds))
                        prob_sg_odds = float(o.iloc[0].get("prob_sg", prob_sg_odds))

                win_prob = 0.6 * win_hist + 0.4 * prob_gol_odds
                cs_prob = 0.4 * cs_hist + 0.6 * prob_sg_odds

                rows.append(
                    TeamMatchupRow(
                        rodada=int(self.rodada),
                        clube_id=int(clube_id),
                        adversario_id=int(adversario_id),
                        mando_casa=int(mando),
                        attack_strength_recent=float(atk_recent),
                        defense_strength_recent=float(def_recent),
                        win_hist_prob=float(win_hist),
                        clean_sheet_hist_prob=float(cs_hist),
                        prob_gol_odds=float(prob_gol_odds),
                        prob_sg_odds=float(prob_sg_odds),
                        attack_multiplier=float(atk_mult),
                        defense_multiplier=float(def_mult),
                        win_probability=float(win_prob),
                        clean_sheet_probability=float(cs_prob),
                        btts_no_probability=0.0,
                    ).__dict__
                )

        df = pd.DataFrame(rows)

        # BTTS-NO aproximado por confronto (usa as duas linhas do jogo)
        for _, grp in df.groupby(["rodada", "clube_id", "adversario_id"]):
            ca = int(grp["clube_id"].iloc[0])
            cb = int(grp["adversario_id"].iloc[0])
            cs_a = grp["clean_sheet_probability"].iloc[0]
            mask_rev = (df["rodada"] == self.rodada) & (df["clube_id"] == cb) & (df["adversario_id"] == ca)
            if mask_rev.any():
                cs_b = df.loc[mask_rev, "clean_sheet_probability"].iloc[0]
            else:
                cs_b = cs_a
            btts_no = cs_a + cs_b - cs_a * cs_b
            df.loc[grp.index, "btts_no_probability"] = float(np.clip(btts_no, 0.0, 1.0))

        logger.info("✅ MatchupAnalyzer: %d linhas de confronto geradas", len(df))
        return df

    @staticmethod
    def apply_to_predictions(predicoes_df: pd.DataFrame, matchups_df: pd.DataFrame) -> pd.DataFrame:
        """Merge por clube_id e aplica multiplicadores táticos em score_final."""
        if "clube_id" not in predicoes_df.columns or matchups_df.empty:
            return predicoes_df

        merged = predicoes_df.merge(
            matchups_df[
                [
                    "clube_id",
                    "attack_multiplier",
                    "defense_multiplier",
                    "win_probability",
                    "clean_sheet_probability",
                ]
            ].drop_duplicates("clube_id"),
            on="clube_id",
            how="left",
        )

        merged["attack_multiplier"] = merged["attack_multiplier"].fillna(1.0)
        merged["defense_multiplier"] = merged["defense_multiplier"].fillna(1.0)
        merged["win_probability"] = merged["win_probability"].fillna(0.5)
        merged["clean_sheet_probability"] = merged["clean_sheet_probability"].fillna(0.3)

        score_base_col = next(
            (c for c in ["score_final", "predicao_ajustada", "predicao"] if c in merged.columns),
            None,
        )
        if score_base_col is None:
            merged["score_cruzado"] = 0.0
            return merged

        def mult_row(row):
            pos = int(row.get("posicao_id", 0))
            base = float(row[score_base_col])
            if pos in (4, 5):  # MEI, ATA
                mult = row["attack_multiplier"] * (0.7 + 0.3 * row["win_probability"])
            elif pos in (1, 2, 3):  # GOL, LAT, ZAG
                mult = row["defense_multiplier"] * (0.7 + 0.3 * row["clean_sheet_probability"])
            else:
                mult = 1.0
            return base * mult

        merged["score_cruzado"] = merged.apply(mult_row, axis=1)
        return merged
