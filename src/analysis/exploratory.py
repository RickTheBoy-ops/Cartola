"""Análise exploratória para identificar padrões ocultos.

Padrões analisados:
- Explosivos em clássicos (+20% em derbys)
- Super regulares (baixo CV, confiáveis)
- Explosivos (alto risco/retorno para capitão)
- Dependentes de mando (+25% em casa)
- Adversários favoritos (destroem times específicos)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class AnalisadorExploratorio:
    """Analisador de padrões temporais e contextuais."""
    
    def __init__(self, config: Dict):
        """Inicializa analisador.
        
        Args:
            config: Configurações do sistema
        """
        self.config = config
        self.min_jogos = config.get('feature_engineering', {}).get('min_jogos_analise', 5)
    
    def analisar_padroes_temporais(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Analisa todos os padrões temporais.
        
        Args:
            df: DataFrame com histórico completo
            
        Returns:
            Dicionário com DataFrames de cada padrão
        """
        return {
            'explosivos_classicos': self._detectar_explosivos_classicos(df),
            'super_regulares': self._detectar_super_regulares(df),
            'explosivos_risco': self._detectar_explosivos_risco(df),
            'dependentes_mando': self._detectar_dependentes_mando(df),
            'adversarios_favoritos': self._detectar_adversarios_favoritos(df)
        }
    
    def _detectar_explosivos_classicos(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detecta jogadores que explodem em clássicos."""
        # Identifica clássicos (pode ser customizado)
        classicos = [
            ('Flamengo', 'Fluminense'),
            ('Flamengo', 'Vasco'),
            ('Flamengo', 'Botafogo'),
            ('Corinthians', 'Palmeiras'),
            ('Corinthians', 'São Paulo'),
            ('Palmeiras', 'São Paulo'),
            ('Grêmio', 'Internacional'),
            ('Atletico-MG', 'Cruzeiro')
        ]
        
        # Marca jogos de clássico
        df['classico'] = False
        for time1, time2 in classicos:
            mask = (
                ((df['time'] == time1) & (df['adversario'] == time2)) |
                ((df['time'] == time2) & (df['adversario'] == time1))
            )
            df.loc[mask, 'classico'] = True
        
        # Agrupa por jogador
        stats = df.groupby('atleta_id').apply(self._calcular_stats_classico).reset_index()
        
        # Filtra jogadores com boost significativo
        explosivos = stats[stats['boost_classico'] > 1.20]
        
        return explosivos.sort_values('boost_classico', ascending=False)
    
    def _calcular_stats_classico(self, group: pd.DataFrame) -> pd.Series:
        """Calcula estatísticas de clássicos para um jogador."""
        classicos = group[group['classico'] == True]
        normais = group[group['classico'] == False]
        
        if len(classicos) < 2 or len(normais) < self.min_jogos:
            return pd.Series({
                'nome': group['nome'].iloc[0] if len(group) > 0 else '',
                'time': group['time'].iloc[0] if len(group) > 0 else '',
                'num_classicos': len(classicos),
                'media_classicos': 0,
                'media_normal': 0,
                'boost_classico': 1.0
            })
        
        media_classicos = classicos['pontos'].mean()
        media_normal = normais['pontos'].mean()
        boost = media_classicos / media_normal if media_normal > 0 else 1.0
        
        return pd.Series({
            'nome': group['nome'].iloc[0],
            'time': group['time'].iloc[0],
            'num_classicos': len(classicos),
            'media_classicos': media_classicos,
            'media_normal': media_normal,
            'boost_classico': boost
        })
    
    def _detectar_super_regulares(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detecta jogadores super regulares (baixo CV)."""
        stats = df.groupby('atleta_id').agg({
            'nome': 'first',
            'time': 'first',
            'posicao': 'first',
            'pontos': ['mean', 'std', 'count']
        }).reset_index()
        
        stats.columns = ['atleta_id', 'nome', 'time', 'posicao', 'media', 'std', 'num_jogos']
        
        # Filtra mínimo de jogos
        stats = stats[stats['num_jogos'] >= self.min_jogos]
        
        # Calcula CV
        stats['cv'] = stats['std'] / stats['media'].replace(0, 1)
        
        # Super regulares: CV < 0.3 e média decente
        regulares = stats[(stats['cv'] < 0.3) & (stats['media'] > 5)]
        
        return regulares.sort_values('cv')
    
    def _detectar_explosivos_risco(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detecta jogadores explosivos (alto risco/retorno)."""
        stats = df.groupby('atleta_id').agg({
            'nome': 'first',
            'time': 'first',
            'posicao': 'first',
            'pontos': ['mean', 'std', 'max', 'count']
        }).reset_index()
        
        stats.columns = ['atleta_id', 'nome', 'time', 'posicao', 'media', 'std', 'max', 'num_jogos']
        
        # Filtra mínimo de jogos
        stats = stats[stats['num_jogos'] >= self.min_jogos]
        
        # Calcula CV
        stats['cv'] = stats['std'] / stats['media'].replace(0, 1)
        
        # Explosivos: CV alto mas com picos grandes
        explosivos = stats[(stats['cv'] > 0.6) & (stats['max'] > 15)]
        
        return explosivos.sort_values('max', ascending=False)
    
    def _detectar_dependentes_mando(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detecta jogadores que rendem muito mais em casa."""
        stats = df.groupby(['atleta_id', 'mando']).agg({
            'nome': 'first',
            'time': 'first',
            'posicao': 'first',
            'pontos': ['mean', 'count']
        }).reset_index()
        
        stats.columns = ['atleta_id', 'mando', 'nome', 'time', 'posicao', 'media', 'num_jogos']
        
        # Pivota para ter casa e fora lado a lado
        pivot = stats.pivot_table(
            index=['atleta_id', 'nome', 'time', 'posicao'],
            columns='mando',
            values=['media', 'num_jogos']
        ).reset_index()
        
        pivot.columns = ['atleta_id', 'nome', 'time', 'posicao', 
                        'media_casa', 'media_fora', 'jogos_casa', 'jogos_fora']
        
        # Filtra mínimo de jogos em ambos contextos
        pivot = pivot[
            (pivot['jogos_casa'] >= 3) & 
            (pivot['jogos_fora'] >= 3)
        ]
        
        # Calcula diferença
        pivot['diferenca'] = pivot['media_casa'] - pivot['media_fora']
        pivot['boost_casa'] = pivot['media_casa'] / pivot['media_fora'].replace(0, 1)
        
        # Dependentes: boost > 25%
        dependentes = pivot[pivot['boost_casa'] > 1.25]
        
        return dependentes.sort_values('boost_casa', ascending=False)
    
    def _detectar_adversarios_favoritos(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detecta combinações jogador-adversário favoritas."""
        stats = df.groupby(['atleta_id', 'adversario']).agg({
            'nome': 'first',
            'time': 'first',
            'pontos': ['mean', 'count']
        }).reset_index()
        
        stats.columns = ['atleta_id', 'adversario', 'nome', 'time', 'media_vs', 'num_jogos']
        
        # Filtra mínimo de jogos
        stats = stats[stats['num_jogos'] >= 3]
        
        # Média geral do jogador
        media_geral = df.groupby('atleta_id')['pontos'].mean().reset_index()
        media_geral.columns = ['atleta_id', 'media_geral']
        
        stats = stats.merge(media_geral, on='atleta_id')
        
        # Calcula boost vs adverssário específico
        stats['boost_vs'] = stats['media_vs'] / stats['media_geral'].replace(0, 1)
        
        # Favoritos: boost > 50%
        favoritos = stats[stats['boost_vs'] > 1.50]
        
        return favoritos.sort_values('boost_vs', ascending=False)
    
    def recomendar_estrategias(self, padroes: Dict[str, pd.DataFrame],
                              jogador_id: int, contexto: Dict) -> List[str]:
        """Recomenda estratégias baseadas em padrões detectados.
        
        Args:
            padroes: Dicionário com padrões detectados
            jogador_id: ID do jogador
            contexto: Contexto do jogo (adversario, mando, etc)
            
        Returns:
            Lista de recomendações
        """
        recomendacoes = []
        
        # Verifica cada padrão
        for tipo, df in padroes.items():
            if jogador_id in df['atleta_id'].values:
                if tipo == 'explosivos_classicos' and contexto.get('classico', False):
                    boost = df[df['atleta_id'] == jogador_id]['boost_classico'].iloc[0]
                    recomendacoes.append(
                        f"🔥 EXPLODE EM CLÁSSICOS! Boost médio: {boost:.1%}"
                    )
                
                elif tipo == 'super_regulares':
                    cv = df[df['atleta_id'] == jogador_id]['cv'].iloc[0]
                    recomendacoes.append(
                        f"✅ SUPER REGULAR! CV: {cv:.2f} (confiável)"
                    )
                
                elif tipo == 'explosivos_risco':
                    max_pts = df[df['atleta_id'] == jogador_id]['max'].iloc[0]
                    recomendacoes.append(
                        f"💥 EXPLOSIVO! Pico: {max_pts:.1f} pts (bom para capitão)"
                    )
                
                elif tipo == 'dependentes_mando' and contexto.get('mando') == 'casa':
                    boost = df[df['atleta_id'] == jogador_id]['boost_casa'].iloc[0]
                    recomendacoes.append(
                        f"🏠 RENDE MUITO EM CASA! Boost: {boost:.1%}"
                    )
                
                elif tipo == 'adversarios_favoritos':
                    adversario = contexto.get('adversario')
                    jogador_df = df[df['atleta_id'] == jogador_id]
                    if adversario in jogador_df['adversario'].values:
                        boost = jogador_df[
                            jogador_df['adversario'] == adversario
                        ]['boost_vs'].iloc[0]
                        recomendacoes.append(
                            f"🎯 DESTRÓI {adversario}! Boost: {boost:.1%}"
                        )
        
        return recomendacoes
