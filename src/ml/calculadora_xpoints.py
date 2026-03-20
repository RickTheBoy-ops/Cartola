"""
Calculadora de xPoints (Expectativa de Pontos por Atleta)

Responsabilidade única: receber dados brutos de atletas + histórico
e devolver um DataFrame com `xpoints` calculado para cada atleta.

Não bate na API. Não exibe escalação. Não otimiza time.
Apenas calcula a expectativa matemática de pontos.

Fórmula do xpoints:
    xpoints = media_ponderada
                x mult_confronto  (Grupo A=1.15 / B=1.00 / C=0.85)
                x norm_h2h        (0.7 + aproveitamento_h2h * 0.6)
                x mult_casa       (1.05 se mandante, 1.0 se visitante)
                x mult_forma      (forma das últimas N rodadas, 0.85–1.15)
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Pesos para média ponderada (mais recente vale mais)
PESOS_HISTORICO = [0.40, 0.25, 0.15, 0.10, 0.05, 0.03, 0.02]

# Multiplicador por tipo de confronto
MULT_CONFRONTO = {'A': 1.15, 'B': 1.00, 'C': 0.85}

# Multiplicador por mando de campo
MULT_CASA = {'mandante': 1.05, 'visitante': 1.00}


class CalculadoraXPoints:
    """
    Calcula expectativa de pontos (xpoints) para cada atleta.

    Parâmetros
    ----------
    atletas : list[dict]
        Atletas já filtrados (somente status_id=7) vindos do RodadaSnapshot.
    historico_df : pd.DataFrame
        Pontuações históricas do banco SQLite (tabela pontuacoes).
    partidas_enriquecidas : pd.DataFrame
        Partidas da rodada com colunas tipo_confronto / score_h2h_a / score_h2h_b.
    mando_por_clube : dict[int, str]
        {clube_id: 'mandante'|'visitante'}
    n_rodadas_forma : int
        Quantas rodadas recentes usar para calcular forma (padrão: 5)
    """

    def __init__(
        self,
        atletas: List[Dict],
        historico_df: pd.DataFrame,
        partidas_enriquecidas: pd.DataFrame,
        mando_por_clube: Optional[Dict[int, str]] = None,
        n_rodadas_forma: int = 5,
    ):
        self.atletas               = atletas
        self.historico_df          = historico_df
        self.partidas_enriquecidas = partidas_enriquecidas
        self.mando_por_clube       = mando_por_clube or {}
        self.n_rodadas_forma       = n_rodadas_forma

        # Pré-computar mapas para performance
        self._mapa_confronto: Dict[int, str]   = {}  # clube_id -> tipo A/B/C
        self._mapa_h2h:       Dict[int, float] = {}  # clube_id -> score H2H
        self._construir_mapas()

    # ----------------------------------------------------------
    # PÚBLICO
    # ----------------------------------------------------------

    def calcular(self) -> pd.DataFrame:
        """
        Retorna DataFrame com uma linha por atleta e as colunas:
            atleta_id, apelido, clube_id, posicao_id, preco,
            media_base, xpoints, mult_confronto, mult_casa,
            mult_forma, norm_h2h, tipo_confronto
        """
        rows = []
        for atleta in self.atletas:
            atleta_id = int(atleta.get('atleta_id', 0))
            clube_id  = int(atleta.get('clube_id', 0))

            media_base    = self._media_ponderada(atleta_id)
            mult_conf     = MULT_CONFRONTO.get(self._mapa_confronto.get(clube_id, 'B'), 1.0)
            h2h_val       = self._mapa_h2h.get(clube_id, 0.5)
            norm_h2h      = 0.7 + h2h_val * 0.6
            mult_casa     = MULT_CASA.get(self.mando_por_clube.get(clube_id, 'visitante'), 1.0)
            mult_forma    = self._mult_forma(atleta_id)

            xpoints = media_base * mult_conf * norm_h2h * mult_casa * mult_forma

            rows.append({
                'atleta_id':      atleta_id,
                'apelido':        atleta.get('apelido', atleta.get('nome', '')),
                'clube_id':       clube_id,
                'posicao_id':     int(atleta.get('posicao_id', 0)),
                'preco':          float(atleta.get('preco_num', atleta.get('preco', 0))),
                'status_id':      int(atleta.get('status_id', 7)),
                'media_base':     round(media_base, 3),
                'mult_confronto': round(mult_conf, 2),
                'mult_casa':      round(mult_casa, 2),
                'mult_forma':     round(mult_forma, 3),
                'norm_h2h':       round(norm_h2h, 3),
                'score_h2h':      round(h2h_val, 3),
                'tipo_confronto': self._mapa_confronto.get(clube_id, 'B'),
                'xpoints':        round(xpoints, 3),
            })

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        df = df[df['preco'] > 0].copy()
        df = df.sort_values('xpoints', ascending=False).reset_index(drop=True)

        logger.info(
            f"🧮 xpoints calculados para {len(df)} atletas | "
            f"máx: {df['xpoints'].max():.2f} | méd: {df['xpoints'].mean():.2f}"
        )
        return df

    # ----------------------------------------------------------
    # PRIVADO
    # ----------------------------------------------------------

    def _construir_mapas(self):
        """Pré-computa tipo de confronto e score H2H por clube."""
        if self.partidas_enriquecidas.empty:
            return
        for _, row in self.partidas_enriquecidas.iterrows():
            ca = int(row.get('clube_id_a', row.get('clube_casa_id', 0)))
            cb = int(row.get('clube_id_b', row.get('clube_visitante_id', 0)))
            tipo = row.get('tipo_confronto', 'B')
            self._mapa_confronto[ca] = tipo
            self._mapa_confronto[cb] = tipo
            self._mapa_h2h[ca] = float(row.get('score_h2h_a', 0.5))
            self._mapa_h2h[cb] = float(row.get('score_h2h_b', 0.5))

    def _historico_atleta(self, atleta_id: int) -> pd.DataFrame:
        if self.historico_df.empty:
            return pd.DataFrame()
        return (
            self.historico_df[self.historico_df['atleta_id'] == atleta_id]
            .sort_values('rodada', ascending=False)
        )

    def _media_ponderada(self, atleta_id: int) -> float:
        """
        Média ponderada das últimas N rodadas.
        Pesos decrescentes: rodada mais recente vale mais.
        Fallback para média simples se histórico insuficiente.
        """
        hist = self._historico_atleta(atleta_id)
        if hist.empty:
            return 0.0

        pontos = hist['pontos'].head(len(PESOS_HISTORICO)).tolist()
        if not pontos:
            return 0.0

        pesos  = PESOS_HISTORICO[:len(pontos)]
        soma_w = sum(pesos)
        return sum(p * w for p, w in zip(pontos, pesos)) / soma_w

    def _mult_forma(self, atleta_id: int) -> float:
        """
        Multiplicador de forma: desempenho relativo à média própria.
        - últimas n_rodadas_forma vs média geral
        - Escala 0.85 (queda de performance) a 1.15 (momento alto)
        """
        hist = self._historico_atleta(atleta_id)
        if len(hist) < self.n_rodadas_forma + 2:
            return 1.0

        media_geral = hist['pontos'].mean()
        if media_geral == 0:
            return 1.0

        media_recente = hist['pontos'].head(self.n_rodadas_forma).mean()
        ratio = media_recente / media_geral

        # Normalizar entre 0.85 e 1.15
        return float(np.clip(0.85 + (ratio - 1.0) * 0.3, 0.85, 1.15))

    @staticmethod
    def mando_por_clube_da_rodada(partidas: List[Dict]) -> Dict[int, str]:
        """
        Utilitário estático: dado a lista de partidas da rodada,
        retorna {clube_id: 'mandante'|'visitante'}.
        """
        mapa = {}
        for p in partidas:
            casa = p.get('clube_casa_id')
            visit = p.get('clube_visitante_id')
            if casa:  mapa[int(casa)]  = 'mandante'
            if visit: mapa[int(visit)] = 'visitante'
        return mapa
