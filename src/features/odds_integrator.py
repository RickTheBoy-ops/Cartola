#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
========================================================================
CARTOLA FC - ODDS INTEGRATOR
========================================================================
Integra probabilidades de casas de apostas como features adicionais.

Como os top cartoleiros usam odds:
  - Probabilidade de gol por atacante/meia
  - Probabilidade de SG (clean sheet) por goleiro/defensor
  - Essas probs servem como "proxy de chance real" do evento acontecer

Fontes suportadas:
  1. The Odds API (gratuito com chave, 500 req/mês free tier)
     → https://the-odds-api.com  (recomendado, tem odds BR)
  2. Fallback via probabilidade implícita pelo histórico interno

Configuração:
  - Adicione ODDS_API_KEY no .env para ativar a fonte externa
  - Sem chave, usa fallback por histórico (calculado internamente)
========================================================================
"""

import os
import logging
import requests
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# MAPEAMENTO: nome do time no Cartola → nome no mercado de apostas
# Ajuste conforme a API de odds utilizada
# -----------------------------------------------------------------------
CLUBE_ODDS_MAP = {
    "Flamengo":         "Flamengo",
    "Palmeiras":        "Palmeiras",
    "Atletico-MG":      "Atletico Mineiro",
    "Botafogo":         "Botafogo",
    "Fluminense":       "Fluminense",
    "Corinthians":      "Corinthians",
    "São Paulo":        "Sao Paulo",
    "Internacional":    "Internacional",
    "Grêmio":           "Gremio",
    "Santos":           "Santos",
    "Vasco":            "Vasco da Gama",
    "Bahia":            "Bahia",
    "Fortaleza":        "Fortaleza",
    "Cruzeiro":         "Cruzeiro",
    "Atlético-PR":      "Atletico Paranaense",
    "Vitória":          "Vitoria",
    "Mirassol":         "Mirassol",
    "RB Bragantino":    "RB Bragantino",
    "Juventude":        "Juventude",
    "Sport":            "Sport Recife",
}

# Posições que se beneficiam de SG (jogo sem sofrer gol)
POSICOES_SG = {1, 2, 3}  # Goleiro, Lateral, Zagueiro

# Posições que se beneficiam de gol/assistência
POSICOES_GOL = {4, 5}  # Meia, Atacante


class OddsIntegrator:
    """
    Integra probabilidades de casas de apostas ao DataFrame de atletas.

    Adiciona as features:
    - prob_gol:    probabilidade de o time marcar pelo menos 1 gol
    - prob_sg:     probabilidade de o time não sofrer gol (clean sheet)
    - odds_score:  score combinado 0-1 para o atleta baseado em posição + odds

    Uso:
        integrator = OddsIntegrator()
        df = integrator.enrich(df, partidas_df, clubes_df)
    """

    THE_ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/soccer_brazil_campeonato/odds"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ODDS_API_KEY", "")
        self._cache: Dict[str, dict] = {}

    # -------------------------------------------------------------------
    # MÉTODO PRINCIPAL
    # -------------------------------------------------------------------

    def enrich(
        self,
        df: pd.DataFrame,
        partidas_df: Optional[pd.DataFrame] = None,
        clubes_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Enriquece o DataFrame de atletas com features de odds.

        Args:
            df:          DataFrame com atletas (precisa de 'clube_id' e 'posicao_id')
            partidas_df: DataFrame com partidas da rodada (precisa de clubes e adversários)
            clubes_df:   DataFrame com mapeamento clube_id → nome

        Returns:
            df com colunas adicionadas: prob_gol, prob_sg, odds_score
        """
        if 'clube_id' not in df.columns or 'posicao_id' not in df.columns:
            logger.warning("⚠️ OddsIntegrator: colunas 'clube_id' ou 'posicao_id' ausentes")
            df['prob_gol'] = 0.5
            df['prob_sg'] = 0.3
            df['odds_score'] = 0.5
            return df

        # Tentar buscar odds externas
        odds_por_clube = {}
        if self.api_key:
            try:
                odds_por_clube = self._fetch_external_odds(clubes_df)
                logger.info(f"📊 Odds externas carregadas para {len(odds_por_clube)} clubes")
            except Exception as e:
                logger.warning(f"⚠️ Falha ao buscar odds externas: {e}. Usando fallback.")

        # Se não conseguiu odds externas, calcula por histórico interno
        if not odds_por_clube:
            odds_por_clube = self._fallback_historical_odds(df, partidas_df)
            logger.info("📊 Usando probabilidades por histórico interno (fallback)")

        # Aplicar odds ao DataFrame
        df = self._apply_odds(df, odds_por_clube)

        return df

    # -------------------------------------------------------------------
    # ODDS EXTERNAS (The Odds API)
    # -------------------------------------------------------------------

    def _fetch_external_odds(self, clubes_df: Optional[pd.DataFrame]) -> Dict[int, dict]:
        """
        Busca odds da API externa e converte para probabilidade implícita.

        Retorna dict: { clube_id: { 'prob_gol': float, 'prob_sg': float } }
        """
        params = {
            "apiKey": self.api_key,
            "regions": "eu",
            "markets": "h2h",
            "oddsFormat": "decimal",
        }

        resp = requests.get(self.THE_ODDS_API_URL, params=params, timeout=15)
        resp.raise_for_status()
        jogos = resp.json()

        odds_map = {}  # nome_time → {'prob_win': float, 'prob_sg': float}

        for jogo in jogos:
            home = jogo.get("home_team", "")
            away = jogo.get("away_team", "")
            bookmakers = jogo.get("bookmakers", [])
            if not bookmakers:
                continue

            # Pegar a primeira casa disponível
            markets = bookmakers[0].get("markets", [])
            h2h = next((m for m in markets if m["key"] == "h2h"), None)
            if not h2h:
                continue

            outcomes = {o["name"]: o["price"] for o in h2h.get("outcomes", [])}
            home_odd = outcomes.get(home, 2.5)
            away_odd = outcomes.get(away, 2.5)
            draw_odd = outcomes.get("Draw", 3.0)

            # Probabilidades implícitas normalizadas
            inv_home = 1 / home_odd
            inv_away = 1 / away_odd
            inv_draw = 1 / draw_odd
            total = inv_home + inv_away + inv_draw

            prob_home_win = inv_home / total
            prob_away_win = inv_away / total

            # prob_gol ≈ probabilidade de vencer (proxy de marcar gol)
            # prob_sg ≈ probabilidade de vencer SEM sofrer (aproximação)
            odds_map[home] = {
                "prob_gol": min(prob_home_win + 0.15, 0.95),   # +15% por pressão de casa
                "prob_sg": prob_home_win * 0.55,
            }
            odds_map[away] = {
                "prob_gol": min(prob_away_win + 0.05, 0.90),
                "prob_sg": prob_away_win * 0.45,
            }

        # Converter nomes para clube_id usando clubes_df ou mapa fixo
        return self._map_names_to_ids(odds_map, clubes_df)

    def _map_names_to_ids(
        self,
        odds_map: Dict[str, dict],
        clubes_df: Optional[pd.DataFrame],
    ) -> Dict[int, dict]:
        """Converte nomes de times para clube_id."""
        result = {}

        if clubes_df is not None and 'nome' in clubes_df.columns and 'id' in clubes_df.columns:
            nome_to_id = dict(zip(clubes_df['nome'], clubes_df['id']))
        else:
            nome_to_id = {}

        # Inverter CLUBE_ODDS_MAP: odds_name → cartola_name
        odds_to_cartola = {v: k for k, v in CLUBE_ODDS_MAP.items()}

        for odds_name, probs in odds_map.items():
            cartola_name = odds_to_cartola.get(odds_name, odds_name)
            clube_id = nome_to_id.get(cartola_name)

            if clube_id:
                result[int(clube_id)] = probs

        return result

    # -------------------------------------------------------------------
    # FALLBACK: PROBABILIDADES POR HISTÓRICO INTERNO
    # -------------------------------------------------------------------

    def _fallback_historical_odds(
        self,
        df: pd.DataFrame,
        partidas_df: Optional[pd.DataFrame],
    ) -> Dict[int, dict]:
        """
        Calcula probabilidades implícitas a partir do histórico interno.

        prob_gol = média de gols marcados / (média de gols marcados + 1)
        prob_sg  = proporção de jogos sem sofrer gol
        """
        odds_map: Dict[int, dict] = {}

        if partidas_df is None or len(partidas_df) == 0:
            # Sem dados de partidas: usar valores neutros
            for clube_id in df['clube_id'].unique():
                odds_map[int(clube_id)] = {'prob_gol': 0.55, 'prob_sg': 0.30}
            return odds_map

        try:
            cols = {'clube_casa_id', 'clube_visitante_id',
                    'placar_oficial_mandante', 'placar_oficial_visitante'}
            if not cols.issubset(partidas_df.columns):
                raise ValueError("Colunas necessárias ausentes em partidas_df")

            # Perspectiva mandante
            home = partidas_df[['clube_casa_id', 'placar_oficial_mandante', 'placar_oficial_visitante']].copy()
            home.columns = ['clube_id', 'gols_marcados', 'gols_sofridos']

            # Perspectiva visitante
            away = partidas_df[['clube_visitante_id', 'placar_oficial_visitante', 'placar_oficial_mandante']].copy()
            away.columns = ['clube_id', 'gols_marcados', 'gols_sofridos']

            historico = pd.concat([home, away], ignore_index=True)

            for clube_id, grp in historico.groupby('clube_id'):
                media_gols = grp['gols_marcados'].mean()
                taxa_sg = (grp['gols_sofridos'] == 0).mean()

                # Sigmoid para converter gols em probabilidade (0.1 → 0.09, 1 → 0.5, 2 → 0.67)
                prob_gol = media_gols / (media_gols + 1.0)
                prob_gol = float(np.clip(prob_gol, 0.05, 0.95))
                prob_sg = float(np.clip(taxa_sg, 0.05, 0.80))

                odds_map[int(clube_id)] = {'prob_gol': prob_gol, 'prob_sg': prob_sg}

        except Exception as e:
            logger.warning(f"⚠️ Erro no fallback histórico: {e}")
            for clube_id in df['clube_id'].unique():
                odds_map[int(clube_id)] = {'prob_gol': 0.55, 'prob_sg': 0.30}

        return odds_map

    # -------------------------------------------------------------------
    # APLICAR ODDS AO DATAFRAME
    # -------------------------------------------------------------------

    def _apply_odds(
        self,
        df: pd.DataFrame,
        odds_por_clube: Dict[int, dict],
    ) -> pd.DataFrame:
        """
        Aplica as probabilidades ao DataFrame de atletas.

        - Atacantes/Meias recebem prob_gol do próprio clube
        - Goleiros/Zagueiros/Laterais recebem prob_sg do próprio clube
        - odds_score combina a probabilidade relevante por posição (0-1)
        """
        df = df.copy()

        df['prob_gol'] = df['clube_id'].map(
            lambda cid: odds_por_clube.get(int(cid), {}).get('prob_gol', 0.5)
        )
        df['prob_sg'] = df['clube_id'].map(
            lambda cid: odds_por_clube.get(int(cid), {}).get('prob_sg', 0.30)
        )

        # odds_score: probabilidade mais relevante por posição
        def calc_odds_score(row):
            pos = int(row.get('posicao_id', 0))
            if pos in POSICOES_GOL:
                return row['prob_gol']
            elif pos in POSICOES_SG:
                return (row['prob_sg'] * 0.6) + (row['prob_gol'] * 0.4)
            else:
                # Técnico e outros: média simples
                return (row['prob_gol'] + row['prob_sg']) / 2

        df['odds_score'] = df.apply(calc_odds_score, axis=1).clip(0, 1)

        logger.info(
            f"🎲 Odds aplicadas: prob_gol_média={df['prob_gol'].mean():.2f} | "
            f"prob_sg_média={df['prob_sg'].mean():.2f}"
        )

        return df
