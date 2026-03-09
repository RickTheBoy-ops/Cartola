"""Sistema de Valorização baseado em research acadêmico.

Fórmulas e estratégias comprovadas:
- Rodada 1: 0.46 × Preço = Pontos mínimos
- Rodada 2: Boost 20% para quem valorizou
- Expulsos: Threshold 30% (alta valorização)
- Pesos dinâmicos por fase do campeonato
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class SistemaValorizacao:
    """Sistema completo de análise de valorização."""
    
    def __init__(self, config: Dict):
        """Inicializa sistema com configurações.
        
        Args:
            config: Dicionário com configurações do sistema
        """
        self.config = config
        self.valorizacao_config = config.get('valorizacao', {})
        self.rodada_1_threshold = self.valorizacao_config.get('rodada_1_threshold', 0.46)
        self.rodada_2_boost = self.valorizacao_config.get('rodada_2_boost', 1.20)
        self.expulso_threshold = self.valorizacao_config.get('expulso_threshold', 0.30)
    
    def calcular_potencial(self, jogador: pd.Series, rodada: int) -> float:
        """Calcula potencial de valorização (0-100).
        
        Args:
            jogador: Série do pandas com dados do jogador
            rodada: Número da rodada atual
            
        Returns:
            Score de valorização de 0-100
        """
        score = 0.0
        
        # Rodada 1: Fórmula 0.46x preço
        if rodada == 1:
            preco = jogador.get('preco_atual', 0)
            pontos_esperados = preco * self.rodada_1_threshold
            media_hist = jogador.get('media_pontos', 0)
            
            if media_hist > pontos_esperados:
                score += 40  # Alta chance de valorizar
        
        # Rodada 2: Boost para quem valorizou
        elif rodada == 2:
            valorizacao_r1 = jogador.get('valorizacao_rodada_1', 0)
            if valorizacao_r1 > 0:
                score += 35
        
        # Jogador voltando de suspensão
        if jogador.get('cartao_vermelho_rodada_anterior', False):
            score += 30  # Preço baixo + provavel valorização
        
        # Jogador expulso na rodada anterior (threshold 30%)
        if jogador.get('expulso_rodada_anterior', False):
            pontuacao_media = jogador.get('media_pontos', 0)
            if pontuacao_media >= self.expulso_threshold * jogador.get('preco_atual', 1):
                score += 25
        
        # Média de valorização histórica
        media_valorizacao = jogador.get('media_valorizacao', 0)
        if media_valorizacao > 0:
            score += min(media_valorizacao * 10, 30)
        
        # Momentum (ultimas 3 rodadas)
        momentum = jogador.get('momentum_3', 0)
        if momentum > jogador.get('media_pontos', 0):
            score += 20
        
        # Regularidade
        cv = jogador.get('cv_pontos', 0)
        if cv < 0.5:  # Baixa variabilidade
            score += 15
        
        return min(score, 100.0)
    
    def estrategia_por_rodada(self, rodada: int) -> Dict[str, float]:
        """Retorna pesos estratégicos dinâmicos por rodada.
        
        Args:
            rodada: Número da rodada
            
        Returns:
            Dicionário com pesos para valorização e pontuação
        """
        if rodada <= 5:
            # Início: Foco em valorização
            return {'valorizacao': 0.70, 'pontuacao': 0.30}
        elif rodada <= 15:
            # Meio: Equilibrado
            return {'valorizacao': 0.50, 'pontuacao': 0.50}
        else:
            # Final: Foco em pontuação
            return {'valorizacao': 0.30, 'pontuacao': 0.70}
    
    def analisar_grupo(self, jogadores_df: pd.DataFrame, rodada: int, 
                       top_n: int = 20) -> pd.DataFrame:
        """Analisa grupo de jogadores e retorna ranking de valorização.
        
        Args:
            jogadores_df: DataFrame com jogadores
            rodada: Rodada atual
            top_n: Número de top jogadores a retornar
            
        Returns:
            DataFrame com top jogadores ordenados por potencial
        """
        # Calcula potencial para cada jogador
        jogadores_df['potencial_valorizacao'] = jogadores_df.apply(
            lambda row: self.calcular_potencial(row, rodada), axis=1
        )
        
        # Pesos da rodada
        pesos = self.estrategia_por_rodada(rodada)
        
        # Score final combinado
        jogadores_df['score_final'] = (
            jogadores_df['potencial_valorizacao'] * pesos['valorizacao'] +
            jogadores_df['pontos_previstos'] * pesos['pontuacao']
        )
        
        # Ordena e retorna top N
        return jogadores_df.nlargest(top_n, 'score_final')
    
    def relatorio_valorizacao(self, analise_df: pd.DataFrame) -> str:
        """Gera relatório formatado de valorização.
        
        Args:
            analise_df: DataFrame com análise de valorização
            
        Returns:
            String com relatório formatado
        """
        relatorio = ["\n" + "="*60]
        relatorio.append("RELATÓRIO DE VALORIZAÇÃO")
        relatorio.append("="*60 + "\n")
        
        for idx, jogador in analise_df.iterrows():
            relatorio.append(f"\n{jogador['nome']}")
            relatorio.append(f"  Posição: {jogador['posicao']} | Time: {jogador['time']}")
            relatorio.append(f"  Preço: C$ {jogador['preco_atual']:.2f}")
            relatorio.append(f"  Potencial Valorização: {jogador['potencial_valorizacao']:.1f}/100")
            relatorio.append(f"  Score Final: {jogador['score_final']:.2f}")
            relatorio.append(f"  Pontos Previstos: {jogador['pontos_previstos']:.2f}")
            relatorio.append("-" * 60)
        
        return "\n".join(relatorio)
