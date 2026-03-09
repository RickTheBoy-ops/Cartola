"""Análise de confrontos baseada em hacks dos top players.

HACKS implementados:
- Goleiro FORA vs fraco: +40% (mais defesas difíceis)
- Lateral com prob SG alta: +35% (chance de SG = 5pts)
- Atacante CASA vs fraco: +30% (mais gols esperados)
- Falso Meia: +50% (preço de meia, pontuação de atacante)
- Volta suspensão: +25% (preço baixo + valorização)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


class AnalisadorConfrontos:
    """Analisador tático de confrontos."""
    
    def __init__(self, config: Dict):
        """Inicializa analisador.
        
        Args:
            config: Dicionário com configurações
        """
        self.config = config
        self.tactical_weights = config.get('tactical_weights', {})
    
    def analisar_confronto(self, time_jogador: str, time_adversario: str,
                          mando: str, historico_df: pd.DataFrame) -> Dict:
        """Analisa confronto e retorna métricas.
        
        Args:
            time_jogador: Time do jogador
            time_adversario: Time adversário
            mando: 'casa' ou 'fora'
            historico_df: DataFrame com histórico
            
        Returns:
            Dicionário com métricas do confronto
        """
        metrics = {}
        
        # Força do time adversário
        forca_adv = self._calcular_forca_time(time_adversario, historico_df)
        metrics['forca_adversario'] = forca_adv
        metrics['adversario_fraco'] = forca_adv < 0.4
        
        # Probabilidade de saldo de gols (SG)
        prob_sg = self._calcular_prob_sg(
            time_jogador, time_adversario, mando, historico_df
        )
        metrics['prob_sg'] = prob_sg
        
        # Gols esperados
        gols_esperados = self._calcular_gols_esperados(
            time_jogador, time_adversario, mando, historico_df
        )
        metrics['gols_esperados'] = gols_esperados
        
        # Histórico do confronto
        confrontos_diretos = historico_df[
            ((historico_df['time'] == time_jogador) & 
             (historico_df['adversario'] == time_adversario)) |
            ((historico_df['time'] == time_adversario) & 
             (historico_df['adversario'] == time_jogador))
        ]
        
        if len(confrontos_diretos) > 0:
            metrics['media_gols_confronto'] = confrontos_diretos['gols'].mean()
            metrics['num_confrontos'] = len(confrontos_diretos)
        else:
            metrics['media_gols_confronto'] = gols_esperados
            metrics['num_confrontos'] = 0
        
        return metrics
    
    def boost_por_posicao(self, posicao: str, confronto_metrics: Dict,
                         jogador_data: pd.Series) -> float:
        """Calcula boost tático por posição.
        
        Args:
            posicao: Posição do jogador
            confronto_metrics: Métricas do confronto
            jogador_data: Dados do jogador
            
        Returns:
            Multiplicador de boost (1.0 = sem boost)
        """
        boost = 1.0
        
        # Goleiro FORA vs fraco
        if posicao == 'gol':
            if (jogador_data.get('mando') == 'fora' and 
                confronto_metrics.get('adversario_fraco', False)):
                boost *= self.tactical_weights.get('goleiro_fora_vs_fraco', 1.40)
        
        # Lateral com prob SG alta
        elif posicao == 'lat':
            if confronto_metrics.get('prob_sg', 0) > 0.4:
                boost *= self.tactical_weights.get('lateral_sg_alto', 1.35)
        
        # Atacante CASA vs fraco
        elif posicao == 'ata':
            if (jogador_data.get('mando') == 'casa' and
                confronto_metrics.get('adversario_fraco', False)):
                boost *= self.tactical_weights.get('atacante_casa_vs_fraco', 1.30)
        
        # Falso Meia (detectado previamente)
        elif posicao == 'mei':
            if jogador_data.get('falso_meia', False):
                boost *= self.tactical_weights.get('falso_meia', 1.50)
        
        # Volta de suspensão (todas posições)
        if jogador_data.get('volta_suspensao', False):
            boost *= self.tactical_weights.get('volta_suspensao', 1.25)
        
        return boost
    
    def _calcular_forca_time(self, time: str, historico_df: pd.DataFrame) -> float:
        """Calcula força relativa do time (0-1)."""
        time_data = historico_df[historico_df['time'] == time]
        
        if len(time_data) == 0:
            return 0.5  # Média se sem dados
        
        # Métricas de força
        media_gols = time_data['gols'].mean()
        media_vitorias = (time_data['resultado'] == 'V').mean()
        media_pontos = time_data['pontos'].mean()
        
        # Normaliza para 0-1
        forca = (media_gols / 3.0 + media_vitorias + media_pontos / 3.0) / 3.0
        return min(max(forca, 0.0), 1.0)
    
    def _calcular_prob_sg(self, time_casa: str, time_fora: str,
                         mando: str, historico_df: pd.DataFrame) -> float:
        """Calcula probabilidade de saldo de gols."""
        # Dados dos times
        casa_data = historico_df[
            (historico_df['time'] == time_casa) & 
            (historico_df['mando'] == 'casa')
        ]
        fora_data = historico_df[
            (historico_df['time'] == time_fora) & 
            (historico_df['mando'] == 'fora')
        ]
        
        if len(casa_data) == 0 or len(fora_data) == 0:
            return 0.3  # Probabilidade base
        
        # Histórico de saldo de gols
        sg_casa = (casa_data['gols'] > casa_data['gols_sofridos']).mean()
        sg_fora = (fora_data['gols'] < fora_data['gols_sofridos']).mean()
        
        # Combina probabilidades
        if mando == 'casa':
            prob = sg_casa * 0.7 + (1 - sg_fora) * 0.3
        else:
            prob = (1 - sg_casa) * 0.3 + sg_fora * 0.7
        
        return prob
    
    def _calcular_gols_esperados(self, time: str, adversario: str,
                                 mando: str, historico_df: pd.DataFrame) -> float:
        """Calcula gols esperados baseado em histórico."""
        time_data = historico_df[
            (historico_df['time'] == time) & 
            (historico_df['mando'] == mando)
        ]
        
        adv_data = historico_df[
            (historico_df['time'] == adversario) & 
            (historico_df['mando'] == ('fora' if mando == 'casa' else 'casa'))
        ]
        
        if len(time_data) == 0:
            return 1.5  # Média geral
        
        media_gols_time = time_data['gols'].mean()
        
        if len(adv_data) > 0:
            media_sofridos_adv = adv_data['gols_sofridos'].mean()
            return (media_gols_time + media_sofridos_adv) / 2.0
        
        return media_gols_time
