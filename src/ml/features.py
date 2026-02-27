import pandas as pd
import numpy as np
from typing import List, Dict

class FeatureEngineer:
    """
    Criação de features avançadas para predição
    """
    
    @staticmethod
    def create_rolling_features(df: pd.DataFrame, windows: List[int] = [3, 5, 10]) -> pd.DataFrame:
        """
        Cria médias móveis e estatísticas rolling
        """
        df = df.sort_values(['atleta_id', 'rodada'])
        
        for window in windows:
            # Média móvel de pontos
            df[f'pontos_media_{window}'] = df.groupby('atleta_id')['pontos'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            
            # Desvio padrão (consistência)
            df[f'pontos_std_{window}'] = df.groupby('atleta_id')['pontos'].transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )
            
            # Máximo recente
            df[f'pontos_max_{window}'] = df.groupby('atleta_id')['pontos'].transform(
                lambda x: x.rolling(window, min_periods=1).max()
            )
            
            # Mínimo recente
            df[f'pontos_min_{window}'] = df.groupby('atleta_id')['pontos'].transform(
                lambda x: x.rolling(window, min_periods=1).min()
            )
        
        return df
    
    @staticmethod
    def create_form_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features de 'momento' do jogador
        """
        df = df.sort_values(['atleta_id', 'rodada'])
        
        # Tendência (últimas 3 rodadas vs últimas 10 rodadas)
        df['tendencia'] = (
            df.groupby('atleta_id')['pontos'].transform(lambda x: x.rolling(3, min_periods=1).mean()) -
            df.groupby('atleta_id')['pontos'].transform(lambda x: x.rolling(10, min_periods=1).mean())
        )
        
        # Contagem de rodadas consecutivas pontuando acima da média
        def get_sequencia(x):
            mask = x['pontos'] > x['media']
            return mask.groupby((~mask).cumsum()).cumsum()

        df['sequencia_positiva'] = df.groupby('atleta_id').apply(get_sequencia).reset_index(level=0, drop=True)
        
        # Pontos por 90 minutos (eficiência)
        df['pontos_por_90min'] = np.where(
            df['minutos_jogados'] > 0,
            (df['pontos'] / df['minutos_jogados']) * 90,
            0
        )
        
        return df
    
    @staticmethod
    def create_opponent_features(df: pd.DataFrame, partidas_df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features relacionadas ao adversário
        """
        if partidas_df is None or len(partidas_df) == 0:
            df['forca_adversario'] = 0.5
            df['mando_casa'] = 1
            return df

        # Merge com informações da partida
        # Primeiro, precisamos saber se o atleta joga em casa ou fora
        # Isso depende do clube_id dele estar no clube_casa_id ou clube_visitante_id
        
        # Simplificação: assume que já temos clube_id no df
        df = df.merge(
            partidas_df[['rodada', 'clube_casa_id', 'clube_visitante_id', 
                        'aproveitamento_mandante', 'aproveitamento_visitante']],
            left_on=['rodada', 'clube_id'],
            right_on=['rodada', 'clube_casa_id'],
            how='left'
        )
        
        # Se não deu match com clube_casa_id, tenta com clube_visitante_id
        df_visitante = df[df['clube_casa_id'].isna()].copy()
        df_casa = df[df['clube_casa_id'].notna()].copy()
        
        if len(df_visitante) > 0:
            df_visitante = df_visitante.drop(columns=['clube_casa_id', 'clube_visitante_id', 'aproveitamento_mandante', 'aproveitamento_visitante'])
            df_visitante = df_visitante.merge(
                partidas_df[['rodada', 'clube_casa_id', 'clube_visitante_id', 
                            'aproveitamento_mandante', 'aproveitamento_visitante']],
                left_on=['rodada', 'clube_id'],
                right_on=['rodada', 'clube_visitante_id'],
                how='left'
            )
        
        df = pd.concat([df_casa, df_visitante])
        
        # Força do adversário (aproveitamento histórico)
        def get_forca(row):
            if pd.isna(row['clube_casa_id']): return 0.5
            if row['clube_id'] == row['clube_casa_id']:
                return row['aproveitamento_visitante']
            else:
                return row['aproveitamento_mandante']

        df['forca_adversario'] = df.apply(get_forca, axis=1)
        
        # Mando de campo (1 = casa, 0 = fora)
        df['mando_casa'] = (df['clube_id'] == df['clube_casa_id']).astype(int)
        
        return df
    
    @staticmethod
    def create_scout_ratios(df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria ratios e proporções de scouts
        """
        # Gols por finalização
        df['gols_por_finalizacao'] = np.where(
            (df['FD'] + df['FT'] + df['FF']) > 0,
            df['G'] / (df['FD'] + df['FT'] + df['FF']),
            0
        )
        
        # Taxa de conversão de assistências
        df['taxa_assistencia'] = np.where(
            df['FS'] > 0,
            df['A'] / df['FS'],
            0
        )
        
        # Eficiência defensiva (desarmes por falta cometida)
        df['eficiencia_defensiva'] = np.where(
            df['FC'] > 0,
            df['DS'] / df['FC'],
            df['DS']  # Se não cometeu faltas, conta apenas desarmes
        )
        
        return df
    
    @staticmethod
    def create_price_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Features relacionadas a preço e valorização
        """
        df = df.sort_values(['atleta_id', 'rodada'])
        
        # Variação de preço acumulada (últimas 5 rodadas)
        df['variacao_acumulada_5'] = df.groupby('atleta_id')['variacao'].transform(
            lambda x: x.rolling(5, min_periods=1).sum()
        )
        
        # Relação preço/pontos (valor)
        df['custo_beneficio'] = np.where(
            df['preco'] > 0,
            df['media'] / df['preco'],
            0
        )
        
        # Pontos necessários para valorizar (MPV - Mínima Pontuação Valorização)
        # Simplificação: MPV ≈ preço * 0.02
        df['mpv'] = df['preco'] * 0.02
        df['distancia_mpv'] = df['media'] - df['mpv']
        
        return df
    
    @classmethod
    def engineer_all_features(
        cls,
        df: pd.DataFrame,
        partidas_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Aplica todas as transformações de features
        """
        df = cls.create_rolling_features(df)
        df = cls.create_form_features(df)
        df = cls.create_opponent_features(df, partidas_df)
        df = cls.create_scout_ratios(df)
        df = cls.create_price_features(df)
        
        # Remove NaN gerados
        df = df.fillna(0)
        
        return df
