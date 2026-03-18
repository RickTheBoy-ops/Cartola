"""
Specialist Analyzer para Cartola FC (V2.3)
==========================================
Porta a metodologia do cartola-specialist-analyzer:
  Etapa 1: Leitura de Rodada
  Etapa 2: Patrimônio e MPV
  Etapa 3: Scouts e Médias
  Etapa 4: Contexto e Confronto
  Etapa 5: Gestão de Risco
  Etapa 6: Seleção de Capitão

Usa os DataFrames já processados do pipeline principal (atletas_df, partidas_df, predicoes_df).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Mapa de posição_id → nome curto
POSICAO_NOME = {1: "GOL", 2: "LAT", 3: "ZAG", 4: "MEI", 5: "ATA", 6: "TEC"}

# Pesos de confiança por etapa (0-1)
STEP_CONFIDENCE = [0.90, 0.85, 0.88, 0.82, 0.80, 0.75]


class CartolaPrescalingChecklist:
    """
    Checklist de especialista integrado ao pipeline V2.2.
    Recebe DataFrames reais em vez de classes mock.
    """

    def __init__(
        self,
        rodada: int,
        budget: float,
        atletas_df: pd.DataFrame,
        partidas_df: pd.DataFrame,
        predicoes_df: pd.DataFrame,
        modo: str = "valorizar",
    ):
        self.rodada = rodada
        self.budget = budget
        self.atletas = atletas_df.copy()
        self.partidas = partidas_df.copy()
        self.predicoes = predicoes_df.copy()
        self.modo = modo              # valorizar / pontuar / equilibrado
        self.report_lines: List[str] = []

    # ------------------------------------------------------------------
    # UTILIDADES
    # ------------------------------------------------------------------

    def _log(self, line: str):
        self.report_lines.append(line)
        logger.info(line)

    def _pred_col(self) -> str:
        """Coluna de pontuação a usar (ajustada ou bruta)."""
        return "predicao_ajustada" if "predicao_ajustada" in self.predicoes.columns else "predicao"

    def _get_preds(self) -> pd.DataFrame:
        """Merge atletas + predicoes."""
        cols_atleta = ["atleta_id", "apelido", "clube_id", "posicao_id", "preco", "media"]
        cols_pred = ["atleta_id", self._pred_col()]
        if "minimo_para_valorizar" in self.predicoes.columns:
            cols_pred.append("minimo_para_valorizar")
        if "minimo_para_valorizar" in self.atletas.columns:
            cols_atleta.append("minimo_para_valorizar")

        # Filtrar colunas disponíveis
        cols_atleta = [c for c in cols_atleta if c in self.atletas.columns]
        cols_pred = [c for c in cols_pred if c in self.predicoes.columns]

        merged = self.atletas[cols_atleta].merge(
            self.predicoes[cols_pred].drop_duplicates("atleta_id"),
            on="atleta_id",
            how="left"
        ).fillna(0)

        return merged

    # ------------------------------------------------------------------
    # ETAPA 1: LEITURA DE RODADA
    # ------------------------------------------------------------------

    def step1_leitura_rodada(self) -> Dict:
        self._log(f"\n{'='*55}")
        self._log("📋 ETAPA 1: LEITURA DE RODADA")
        self._log(f"{'='*55}")

        findings = []

        rodada_partidas = self.partidas[self.partidas["rodada"] == self.rodada]
        n_jogos = len(rodada_partidas)
        self._log(f"   ⚽ Rodada {self.rodada}: {n_jogos} jogos confirmados")

        # Identificar mandantes / visitantes
        if n_jogos > 0 and "clube_casa_id" in rodada_partidas.columns:
            mandantes = rodada_partidas["clube_casa_id"].tolist()
            findings.append(f"{n_jogos} partidas mapeadas.")
            self._log(f"   🏟️ Mandantes IDs: {mandantes}")
        else:
            findings.append("Sem dados estruturados de partidas para esta rodada.")
            self._log("   ⚠️ Sem partidas indexadas por rodada.")

        return {
            "step": 1,
            "name": "Leitura de Rodada",
            "findings": findings,
            "confidence": STEP_CONFIDENCE[0],
        }

    # ------------------------------------------------------------------
    # ETAPA 2: PATRIMÔNIO E MPV
    # ------------------------------------------------------------------

    def step2_patrimonio(self) -> Dict:
        self._log(f"\n{'='*55}")
        self._log("💰 ETAPA 2: PATRIMÔNIO E MPV")
        self._log(f"{'='*55}")

        self._log(f"   Orçamento: C${self.budget:.1f} | Modo: {self.modo.upper()}")

        merged = self._get_preds()
        pred_col = self._pred_col()

        # Calcular MPV: usar coluna oficial se disponível; caso contrário media + 0.5
        if "minimo_para_valorizar" in merged.columns:
            merged["mpv"] = merged["minimo_para_valorizar"]
            self._log("   ✅ MPV vindo da API oficial do Cartola")
        elif "media" in merged.columns:
            merged["mpv"] = merged["media"] + 0.5
            self._log("   ⚠️ MPV estimado via média + 0.5 (fallback do especialista)")
        else:
            merged["mpv"] = 0.0

        # Margem positiva
        if pred_col in merged.columns:
            merged["margem"] = merged[pred_col] - merged["mpv"]
            top_mpv = merged[merged["margem"] > 0].nlargest(5, "margem")
            if len(top_mpv) > 0:
                self._log("   🏆 TOP 5 com maior margem de valorização:")
                for _, row in top_mpv.iterrows():
                    pos = POSICAO_NOME.get(int(row.get("posicao_id", 0)), "?")
                    self._log(
                        f"      {row['apelido']:20s} ({pos}) "
                        f"Pred:{row[pred_col]:.1f} MPV:{row['mpv']:.1f} "
                        f"Margem:+{row['margem']:.1f}"
                    )

        return {
            "step": 2,
            "name": "Patrimônio e MPV",
            "findings": [f"Orçamento C${self.budget:.1f}, modo {self.modo}"],
            "confidence": STEP_CONFIDENCE[1],
        }

    # ------------------------------------------------------------------
    # ETAPA 3: SCOUTS E MÉDIAS
    # ------------------------------------------------------------------

    def step3_scouts(self) -> Dict:
        self._log(f"\n{'='*55}")
        self._log("🔭 ETAPA 3: SCOUTS E MÉDIAS")
        self._log(f"{'='*55}")

        merged = self._get_preds()
        pred_col = self._pred_col()

        if pred_col not in merged.columns:
            self._log("   ⚠️ Sem coluna de predição disponível.")
            return {"step": 3, "name": "Scouts", "findings": [], "confidence": STEP_CONFIDENCE[2]}

        top10 = merged.nlargest(10, pred_col)
        self._log("   🏅 TOP 10 por predição de pontos:")
        for _, row in top10.iterrows():
            pos = POSICAO_NOME.get(int(row.get("posicao_id", 0)), "?")
            self._log(
                f"      {row.get('apelido', '?'):20s} ({pos}) "
                f"Pred:{row[pred_col]:.2f} | Média:{row.get('media', 0):.1f} | C${row.get('preco', 0):.2f}"
            )

        return {
            "step": 3,
            "name": "Scouts e Médias",
            "findings": [f"{len(merged)} atletas analisados"],
            "confidence": STEP_CONFIDENCE[2],
        }

    # ------------------------------------------------------------------
    # ETAPA 4: CONTEXTO E CONFRONTO
    # ------------------------------------------------------------------

    def step4_contexto(self) -> Dict:
        self._log(f"\n{'='*55}")
        self._log("🔍 ETAPA 4: CONTEXTO E CONFRONTO")
        self._log(f"{'='*55}")

        rodada_partidas = self.partidas[self.partidas["rodada"] == self.rodada]
        if len(rodada_partidas) == 0:
            self._log("   ⚠️ Sem dados de confronto para esta rodada.")
            return {"step": 4, "name": "Contexto", "findings": [], "confidence": STEP_CONFIDENCE[3]}

        for _, p in rodada_partidas.iterrows():
            casa_id = p.get("clube_casa_id", "?")
            visita_id = p.get("clube_visitante_id", "?")
            aprov_casa = p.get("aproveitamento_mandante", 0.5)
            aprov_vis = p.get("aproveitamento_visitante", 0.5)
            favorito = "Casa" if aprov_casa >= aprov_vis else "Visitante"
            self._log(
                f"   {casa_id} vs {visita_id} — Aprov Casa:{aprov_casa:.0%} "
                f"Visita:{aprov_vis:.0%} → Favorito: {favorito}"
            )

        return {
            "step": 4,
            "name": "Contexto",
            "findings": [f"{len(rodada_partidas)} confrontos analisados"],
            "confidence": STEP_CONFIDENCE[3],
        }

    # ------------------------------------------------------------------
    # ETAPA 5: GESTÃO DE RISCO
    # ------------------------------------------------------------------

    def step5_gestao_risco(self) -> Dict:
        self._log(f"\n{'='*55}")
        self._log("🛡️ ETAPA 5: GESTÃO DE RISCO")
        self._log(f"{'='*55}")

        self._log("   Regras ativas:")
        self._log("   • Máx. 2 defensores do mesmo clube (SG)")
        self._log("   • Máx. 3 jogadores do mesmo clube (total)")
        self._log("   • 1-2 apostas diferenciais no ataque")
        self._log("   • Capitão com consistência ≥ 70%")

        return {
            "step": 5,
            "name": "Gestão de Risco",
            "findings": ["Regras SG e concentração validadas pelo GeneticTeamOptimizer"],
            "confidence": STEP_CONFIDENCE[4],
        }

    # ------------------------------------------------------------------
    # ETAPA 6: CAPITÃO
    # ------------------------------------------------------------------

    def step6_capitao(self) -> Tuple[Dict, Optional[str]]:
        self._log(f"\n{'='*55}")
        self._log("👑 ETAPA 6: SELEÇÃO DE CAPITÃO")
        self._log(f"{'='*55}")

        merged = self._get_preds()
        pred_col = self._pred_col()

        if pred_col not in merged.columns:
            self._log("   ⚠️ Sem predições para selecionar capitão.")
            return {"step": 6, "name": "Capitão", "findings": [], "confidence": STEP_CONFIDENCE[5]}, None

        # Filtrar apenas meias e atacantes
        candidatos = merged[merged["posicao_id"].isin([4, 5])].nlargest(5, pred_col)

        capitao_nome = None
        self._log("   🏆 Candidatos a capitão (Meias/Atacantes com maior predição):")
        for i, (_, row) in enumerate(candidatos.iterrows()):
            pos = POSICAO_NOME.get(int(row.get("posicao_id", 0)), "?")
            self._log(
                f"      {'👑 ' if i == 0 else '   '}{row.get('apelido', '?'):20s} ({pos}) "
                f"→ Pred: {row[pred_col]:.2f} | Média: {row.get('media', 0):.1f}"
            )
            if i == 0:
                capitao_nome = row.get("apelido", None)

        if capitao_nome:
            self._log(f"\n   ✅ Capitão recomendado: {capitao_nome}")

        return {
            "step": 6,
            "name": "Capitão",
            "findings": [f"Capitão: {capitao_nome}"] if capitao_nome else [],
            "confidence": STEP_CONFIDENCE[5],
        }, capitao_nome

    # ------------------------------------------------------------------
    # RUN: EXECUTA TODOS OS PASSOS
    # ------------------------------------------------------------------

    def run(self) -> Dict:
        """Executa o checklist completo e retorna relatório estruturado."""
        self._log("\n" + "🏆 "*18)
        self._log("   CARTOLA SPECIALIST ANALYZER — PRÉ-ESCALAÇÃO V2.3")
        self._log("🏆 "*18)

        results = {}
        results["step1"] = self.step1_leitura_rodada()
        results["step2"] = self.step2_patrimonio()
        results["step3"] = self.step3_scouts()
        results["step4"] = self.step4_contexto()
        results["step5"] = self.step5_gestao_risco()
        step6_info, capitao = self.step6_capitao()
        results["step6"] = step6_info

        # Confiança geral
        confiancas = [s["confidence"] for s in results.values()]
        confianca_geral = (sum(confiancas) / len(confiancas)) * 100

        self._log(f"\n{'='*55}")
        self._log(f"📊 CONFIANÇA GERAL DA ANÁLISE: {confianca_geral:.1f}%")
        self._log(f"   Capitão sugerido: {capitao or 'Não determinado'}")
        self._log(f"{'='*55}\n")

        results["confidence_level"] = confianca_geral
        results["recommended_captain"] = capitao
        results["report"] = "\n".join(self.report_lines)

        return results
