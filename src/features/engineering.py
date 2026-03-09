"""Feature Engineering baseado em papers acadêmicos.

Features comprovadas (papers UFU 2025, IFRS 2024):
1. Momentum (3 rodadas) - Melhor preditor
2. Explosão (desvio vs média)
3. Scouts ponderados por posição
4. Regularidade (CV inverso)
5. Favoristmo (mando + adverssário)
6. Interações (features compostas)
"""

import pandas as pd
import numpy as np
from typing import Dict, List


class FeatureEngineer:
    """Engine de feature engineering avançado."""
    
    def __init__(self, config: Dict):
        """Inicializa engine.
        
        Args:
            config: Configurações do sistema
        """
        self.config = config
        self.fe_config = config.get('feature_engineering', {})
        self.rolling_windows = self.fe_config.get('rolling_windows', [3, 5, 10])
        self.ewm_spans = self.fe_config.get('ewm_spans', [3, 5])
        
        # Pesos de scouts por posição (baseado em research)
        self.scout_weights = {
            'gol': {'defesa': 1.0, 'gol_contra': -1.5, 'sg': 1.2},
            'zag': {'defesa': 0.8, 'gol': 1.5, 'sg': 1.0, 'desarme': 0.5},
            'lat': {'defesa': 0.7, 'assistencia': 1.2, 'sg': 1.0, 'cruzamento': 0.4},
            'mei': {'assistencia': 1.0, 'gol': 1.3, 'finalizacao': 0.6, 'passe': 0.3},
            'ata': {'gol': 1.5, 'finalizacao': 0.8, 'assistencia': 0.9}
        }
    
    def criar_features_avancadas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cria todas as features avançadas.
        
        Args:
            df: DataFrame com dados brutos
            
        Returns:
            DataFrame enriquecido com features
        """
        df = df.copy()
        
        # 1. Momentum (rolling means)
        df = self._add_momentum_features(df)
        
        # 2. Explosão (desvio recente vs histórico)
        df = self._add_explosao_features(df)
        
        # 3. Scouts ponderados
        df = self._add_scouts_ponderados(df)
        
        # 4. Regularidade
        df = self._add_regularidade_features(df)
        
        # 5. Favoristmo (contexto do jogo)
        df = self._add_favoristmo_features(df)
        
        # 6. Interações
        df = self._add_interacoes(df)
        
        # 7. Exponential weighted moving average
        df = self._add_ewm_features(df)
        
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features de momentum (rolling means)."""
        for window in self.rolling_windows:
            df[f'momentum_{window}'] = df.groupby('atleta_id')['pontos'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
        return df
    
    def _add_explosao_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features de explosão."""
        # Média histórica
        df['media_historica'] = df.groupby('atleta_id')['pontos'].transform('mean')
        
        # Desvio recente (3 rodadas) vs histórica
        df['momentum_3'] = df.groupby('atleta_id')['pontos'].transform(
            lambda x: x.rolling(3, min_periods=1).mean()
        )
        df['explosao'] = df['momentum_3'] - df['media_historica']
        df['explosao_pct'] = (df['explosao'] / df['media_historica'].replace(0, 1)) * 100
        
        return df
    
    def _add_scouts_ponderados(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona scouts ponderados por posição."""
        def calcular_scout_ponderado(row):
            posicao = row.get('posicao', 'mei')
            weights = self.scout_weights.get(posicao, {})
            
            score = 0
            for scout, weight in weights.items():
                scout_value = row.get(scout, 0)
                score += scout_value * weight
            
            return score
        
        df['scouts_ponderado'] = df.apply(calcular_scout_ponderado, axis=1)
        return df
    
    def _add_regularidade_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features de regularidade (inverso do CV)."""
        # Coeficiente de variação
        df['std_pontos'] = df.groupby('atleta_id')['pontos'].transform('std')
        df['cv_pontos'] = df['std_pontos'] / df['media_historica'].replace(0, 1)
        
        # Regularidade = inverso do CV (normalizado)
        df['regularidade'] = 1 / (1 + df['cv_pontos'])
        
        return df
    
    def _add_favoristmo_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features de favoristmo (contexto)."""
        # Mando (1 = casa, 0 = fora)
        df['mando_bin'] = (df['mando'] == 'casa').astype(int)
        
        # Força do adversário (se disponível)
        if 'forca_adversario' in df.columns:
            df['favoristmo'] = df['mando_bin'] * 0.3 + (1 - df['forca_adversario']) * 0.7
        else:
            df['favoristmo'] = df['mando_bin']  # Apenas mando
        
        return df
    
    def _add_interacoes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona features de interação."""
        # Momentum × Favoristmo
        df['expectativa'] = df['momentum_3'] * df['favoristmo']
        
        # Regularidade × Momentum
        df['confianca'] = df['regularidade'] * df['momentum_3']
        
        # Preço × Média (value for money)
        df['value_score'] = df['media_historica'] / df['preco_atual'].replace(0, 1)
        
        return df
    
    def _add_ewm_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona exponential weighted moving average."""
        for span in self.ewm_spans:
            df[f'ewm_{span}'] = df.groupby('atleta_id')['pontos'].transform(
                lambda x: x.ewm(span=span, adjust=False).mean()
            )
        return df
    
    def selecionar_features_modelo(self, df: pd.DataFrame) -> List[str]:
        """Seleciona features para treino de modelo.
        
        Args:
            df: DataFrame com todas features
            
        Returns:
            Lista de nomes de features selecionadas
        """
        features = [
            # Momentum
            'momentum_3', 'momentum_5', 'momentum_10',
            
            # Explosão
            'explosao', 'explosao_pct',
            
            # Scouts
            'scouts_ponderado',
            
            # Regularidade
            'regularidade', 'cv_pontos',
            
            # Contexto
            'favoristmo', 'mando_bin',
            
            # Interações
            'expectativa', 'confianca', 'value_score',
            
            # EWM
            'ewm_3', 'ewm_5',
            
            # Básicas
            'media_historica', 'preco_atual'
        ]
        
        # Filtra apenas features que existem no DataFrame
        return [f for f in features if f in df.columns]
