#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
========================================================================
CARTOLA FC - FEATURE ENGINEERING V2
========================================================================
Engine de features com análise estatística por posição.

Refatorado de: cartola_v2_feature_eng.py

Melhorias:
  - Percentis por posição (LAT, ZAG, GOL têm métricas diferentes)
  - Normalização estatística (Z-score adaptativo)
  - Detecção automática de outliers
  - Sistema de pesos dinâmicos
========================================================================
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Set
import warnings

warnings.filterwarnings('ignore')


class FeatureEngineeringV2:
    """
    Engine de Feature Engineering com análise por posição.
    
    Cria 'mega_score' otimizado para cada jogador.
    """
    
    # Posições disponíveis no Cartola
    POSITIONS = {
        1: 'GOL',
        2: 'LAT',
        3: 'ZAG',
        4: 'MEI',
        5: 'ATA',
        6: 'TEC'
    }
    
    # Features base para análise
    BASE_FEATURES = [
        'media',
        'pontos_ultimas_5',
        'variancia',
        'jogos',
        'minutos_jogados'
    ]
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: Configurações personalizadas
        """
        
        self.default_config = {
            'min_games': 3,
            'use_percentiles': True,
            'use_zscore_normalization': True,
            'remove_outliers': True,
            'outlier_threshold': 3.0,
            'weights': {
                'GOL': {
                    'media': 0.40,
                    'pontos_ultimas_5': 0.30,
                    'variancia': -0.15,
                    'jogos': 0.10,
                    'minutos_jogados': 0.15
                },
                'LAT': {
                    'media': 0.35,
                    'pontos_ultimas_5': 0.35,
                    'variancia': -0.10,
                    'jogos': 0.10,
                    'minutos_jogados': 0.10
                },
                'ZAG': {
                    'media': 0.40,
                    'pontos_ultimas_5': 0.30,
                    'variancia': -0.15,
                    'jogos': 0.10,
                    'minutos_jogados': 0.15
                },
                'MEI': {
                    'media': 0.30,
                    'pontos_ultimas_5': 0.40,
                    'variancia': -0.05,
                    'jogos': 0.10,
                    'minutos_jogados': 0.15
                },
                'ATA': {
                    'media': 0.30,
                    'pontos_ultimas_5': 0.45,
                    'variancia': -0.05,
                    'jogos': 0.05,
                    'minutos_jogados': 0.15
                },
                'TEC': {
                    'media': 0.35,
                    'pontos_ultimas_5': 0.35,
                    'variancia': -0.10,
                    'jogos': 0.10,
                    'minutos_jogados': 0.10
                }
            }
        }
        
        self.config = {**self.default_config, **(config or {})}
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features avançadas para o DataFrame.
        
        Args:
            df: DataFrame com dados brutos dos jogadores
            
        Returns:
            DataFrame com novas features incluíndo 'mega_score'
        """
        
        df = df.copy()
        
        print("\n🧠 Feature Engineering V2 - Iniciando...")
        
        # 1. Criar features básicas
        df = self._create_base_features(df)
        
        # 2. Normalizar features por posição
        if self.config['use_zscore_normalization']:
            df = self._normalize_by_position(df)
        
        # 3. Calcular percentis por posição
        if self.config['use_percentiles']:
            df = self._calculate_percentiles(df)
        
        # 4. Remover outliers
        if self.config['remove_outliers']:
            df = self._remove_outliers(df)
        
        # 5. Calcular mega_score final
        df = self._calculate_mega_score(df)
        
        print(f"   ✅ Features criadas! Mega_score médio: {df['mega_score'].mean():.2f}")
        
        return df
    
    def _create_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features básicas derivadas.
        """
        
        # Variância recente (estabilidade)
        if 'pontos_ultimas_5' in df.columns:
            df['variancia'] = df['pontos_ultimas_5'].rolling(
                window=3, min_periods=1
            ).std().fillna(0)
        else:
            df['variancia'] = 0
        
        # Minutos jogados (disponibilidade)
        if 'minutos_jogados' not in df.columns:
            df['minutos_jogados'] = df['jogos'] * 90  # Aproximação
        
        # Pontuação por minuto
        df['pontos_por_minuto'] = np.where(
            df['minutos_jogados'] > 0,
            df['media'] / (df['minutos_jogados'] / 90),
            0
        )
        
        return df
    
    def _normalize_by_position(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normaliza features usando Z-score POR POSIÇÃO.
        
        Beneficia comparação justa entre posições.
        """
        
        for pos_id, pos_name in self.POSITIONS.items():
            mask = df['posicao_id'] == pos_id
            
            if mask.sum() == 0:
                continue
            
            for feature in self.BASE_FEATURES:
                if feature not in df.columns:
                    continue
                
                col_name = f"{feature}_norm"
                
                # Calcular Z-score dentro da posição
                mean = df.loc[mask, feature].mean()
                std = df.loc[mask, feature].std()
                
                if std > 0:
                    df.loc[mask, col_name] = (
                        (df.loc[mask, feature] - mean) / std
                    )
                else:
                    df.loc[mask, col_name] = 0
        
        return df
    
    def _calculate_percentiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula percentis por posição para cada feature.
        
        Facilita identificação de jogadores top.
        """
        
        for pos_id, pos_name in self.POSITIONS.items():
            mask = df['posicao_id'] == pos_id
            
            if mask.sum() == 0:
                continue
            
            for feature in self.BASE_FEATURES:
                if feature not in df.columns:
                    continue
                
                col_name = f"{feature}_pct"
                
                # Calcular percentil dentro da posição
                df.loc[mask, col_name] = df.loc[mask, feature].rank(
                    pct=True
                ) * 100
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove outliers usando threshold de Z-score.
        
        Previne distorções causadas por valores extremos.
        """
        
        threshold = self.config['outlier_threshold']
        initial_count = len(df)
        
        for feature in self.BASE_FEATURES:
            norm_col = f"{feature}_norm"
            
            if norm_col not in df.columns:
                continue
            
            # Remover linhas com Z-score extremo
            df = df[np.abs(df[norm_col]) <= threshold]
        
        removed = initial_count - len(df)
        
        if removed > 0:
            print(f"   ⚠️ Outliers removidos: {removed} jogadores")
        
        return df
    
    def _calculate_mega_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula mega_score final usando pesos por posição.
        
        Fórmula:
            mega_score = SUM(weight_i * feature_norm_i)
        """
        
        df['mega_score'] = 0.0
        
        for pos_id, pos_name in self.POSITIONS.items():
            mask = df['posicao_id'] == pos_id
            
            if mask.sum() == 0:
                continue
            
            weights = self.config['weights'].get(pos_name, {})
            
            score = 0
            
            for feature, weight in weights.items():
                norm_col = f"{feature}_norm"
                
                if norm_col in df.columns:
                    score += df.loc[mask, norm_col] * weight
            
            df.loc[mask, 'mega_score'] = score
        
        # Normalizar mega_score para escala 0-100
        if df['mega_score'].std() > 0:
            df['mega_score'] = (
                (df['mega_score'] - df['mega_score'].min()) / 
                (df['mega_score'].max() - df['mega_score'].min())
            ) * 100
        
        return df
    
    def get_top_players(self, df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
        """
        Retorna os N melhores jogadores por mega_score.
        
        Args:
            df: DataFrame com mega_score calculado
            n: Quantidade de jogadores
            
        Returns:
            DataFrame ordenado
        """
        
        if 'mega_score' not in df.columns:
            raise ValueError("DataFrame não possui 'mega_score'. Execute engineer_features() primeiro.")
        
        return df.nlargest(n, 'mega_score')
    
    def get_position_stats(self, df: pd.DataFrame) -> Dict:
        """
        Retorna estatísticas de mega_score por posição.
        
        Returns:
            Dicionário com médias por posição
        """
        
        if 'mega_score' not in df.columns:
            raise ValueError("DataFrame não possui 'mega_score'.")
        
        stats = {}
        
        for pos_id, pos_name in self.POSITIONS.items():
            mask = df['posicao_id'] == pos_id
            
            if mask.sum() > 0:
                stats[pos_name] = {
                    'count': mask.sum(),
                    'mean_score': df.loc[mask, 'mega_score'].mean(),
                    'max_score': df.loc[mask, 'mega_score'].max(),
                    'min_score': df.loc[mask, 'mega_score'].min()
                }
        
        return stats
