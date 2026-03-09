"""Detectores de padrões especiais baseados em hacks.

Padrões detectados:
- Falsos Meias: Meias com perfil de atacante (OURO)
- Laterais Ofensivos: Muitas assistências + SG possível
- Zagueiros Artilheiros: Marcam gols em escanteio
- Voltando de Suspensão: Preço baixo + alta valorização
"""

import pandas as pd
import numpy as np
from typing import Dict, List


class DetectorPadroesEspeciais:
    """Detector de jogadores com padrões especiais."""
    
    def __init__(self, config: Dict):
        """Inicializa detector.
        
        Args:
            config: Configurações do sistema
        """
        self.config = config
    
    def detectar_falsos_meias(self, df: pd.DataFrame) -> List[Dict]:
        """Detecta meias com perfil de atacante.
        
        Critérios:
        - Mais de 3 finalizações por jogo
        - Mais de 0.3 gols por jogo
        - Taxa de conversão > 10%
        
        Args:
            df: DataFrame com dados dos jogadores
            
        Returns:
            Lista de dicionários com jogadores detectados
        """
        meias = df[df['posicao'] == 'mei'].copy()
        
        if len(meias) == 0:
            return []
        
        # Agrupa por jogador
        stats = meias.groupby('atleta_id').agg({
            'nome': 'first',
            'time': 'first',
            'preco_atual': 'last',
            'finalizacao': 'mean',
            'gol': 'mean',
            'pontos': 'mean'
        }).reset_index()
        
        # Aplica critérios
        stats['taxa_conversao'] = (stats['gol'] / stats['finalizacao'].replace(0, 1)) * 100
        
        falsos_meias = stats[
            (stats['finalizacao'] > 3) & 
            (stats['gol'] > 0.3) & 
            (stats['taxa_conversao'] > 10)
        ]
        
        result = []
        for _, jogador in falsos_meias.iterrows():
            result.append({
                'atleta_id': jogador['atleta_id'],
                'nome': jogador['nome'],
                'time': jogador['time'],
                'preco': jogador['preco_atual'],
                'finalizacoes_media': jogador['finalizacao'],
                'gols_media': jogador['gol'],
                'taxa_conversao': jogador['taxa_conversao'],
                'pontos_media': jogador['pontos'],
                'tipo': 'FALSO_MEIA'
            })
        
        return result
    
    def detectar_laterais_ofensivos(self, df: pd.DataFrame) -> List[Dict]:
        """Detecta laterais com alto potencial ofensivo.
        
        Critérios:
        - Mais de 0.2 assistências por jogo
        - Bom histórico de SG
        
        Args:
            df: DataFrame com dados dos jogadores
            
        Returns:
            Lista de jogadores detectados
        """
        laterais = df[df['posicao'] == 'lat'].copy()
        
        if len(laterais) == 0:
            return []
        
        stats = laterais.groupby('atleta_id').agg({
            'nome': 'first',
            'time': 'first',
            'preco_atual': 'last',
            'assistencia': 'mean',
            'sg': 'mean',
            'cruzamento': 'mean',
            'pontos': 'mean'
        }).reset_index()
        
        ofensivos = stats[
            (stats['assistencia'] > 0.2) | 
            (stats['sg'] > 0.3)
        ]
        
        result = []
        for _, jogador in ofensivos.iterrows():
            result.append({
                'atleta_id': jogador['atleta_id'],
                'nome': jogador['nome'],
                'time': jogador['time'],
                'preco': jogador['preco_atual'],
                'assistencias_media': jogador['assistencia'],
                'sg_media': jogador['sg'],
                'cruzamentos_media': jogador['cruzamento'],
                'pontos_media': jogador['pontos'],
                'tipo': 'LATERAL_OFENSIVO'
            })
        
        return result
    
    def detectar_zagueiros_artilheiros(self, df: pd.DataFrame) -> List[Dict]:
        """Detecta zagueiros que marcam gols.
        
        Critérios:
        - Mais de 0.15 gols por jogo
        - Bom aproveitamento em bolas paradas
        
        Args:
            df: DataFrame com dados
            
        Returns:
            Lista de jogadores
        """
        zagueiros = df[df['posicao'] == 'zag'].copy()
        
        if len(zagueiros) == 0:
            return []
        
        stats = zagueiros.groupby('atleta_id').agg({
            'nome': 'first',
            'time': 'first',
            'preco_atual': 'last',
            'gol': 'mean',
            'sg': 'mean',
            'pontos': 'mean'
        }).reset_index()
        
        artilheiros = stats[stats['gol'] > 0.15]
        
        result = []
        for _, jogador in artilheiros.iterrows():
            result.append({
                'atleta_id': jogador['atleta_id'],
                'nome': jogador['nome'],
                'time': jogador['time'],
                'preco': jogador['preco_atual'],
                'gols_media': jogador['gol'],
                'sg_media': jogador['sg'],
                'pontos_media': jogador['pontos'],
                'tipo': 'ZAGUEIRO_ARTILHEIRO'
            })
        
        return result
    
    def detectar_voltando_suspensao(self, df: pd.DataFrame, 
                                    rodada_atual: int) -> List[Dict]:
        """Detecta jogadores voltando de suspensão.
        
        Oportunidade: Preço baixo + provavel valorização
        
        Args:
            df: DataFrame com dados
            rodada_atual: Número da rodada atual
            
        Returns:
            Lista de jogadores
        """
        if 'cartao_vermelho' not in df.columns:
            return []
        
        # Jogadores com cartão vermelho na rodada anterior
        rodada_anterior = df[df['rodada'] == rodada_atual - 1]
        suspensos = rodada_anterior[rodada_anterior['cartao_vermelho'] == 1]
        
        result = []
        for _, jogador in suspensos.iterrows():
            result.append({
                'atleta_id': jogador['atleta_id'],
                'nome': jogador['nome'],
                'time': jogador['time'],
                'posicao': jogador['posicao'],
                'preco': jogador['preco_atual'],
                'media_antes_suspensao': jogador['media_pontos'],
                'tipo': 'VOLTA_SUSPENSAO'
            })
        
        return result
    
    def executar_todas_deteccoes(self, df: pd.DataFrame, 
                                 rodada_atual: int) -> Dict[str, List]:
        """Executa todas as detecções.
        
        Args:
            df: DataFrame com dados
            rodada_atual: Rodada atual
            
        Returns:
            Dicionário com resultados de todas detecções
        """
        return {
            'falsos_meias': self.detectar_falsos_meias(df),
            'laterais_ofensivos': self.detectar_laterais_ofensivos(df),
            'zagueiros_artilheiros': self.detectar_zagueiros_artilheiros(df),
            'volta_suspensao': self.detectar_voltando_suspensao(df, rodada_atual)
        }
    
    def gerar_relatorio_deteccoes(self, deteccoes: Dict[str, List]) -> str:
        """Gera relatório formatado das detecções.
        
        Args:
            deteccoes: Resultados das detecções
            
        Returns:
            String com relatório
        """
        relatorio = ["\n" + "="*70]
        relatorio.append("DETECÇÕES DE PADRÕES ESPECIAIS")
        relatorio.append("="*70 + "\n")
        
        # Falsos Meias
        if deteccoes['falsos_meias']:
            relatorio.append("\n🥇 FALSOS MEIAS (OURO!)")
            relatorio.append("-" * 70)
            for j in deteccoes['falsos_meias']:
                relatorio.append(f"  {j['nome']} ({j['time']})")
                relatorio.append(f"    Preço: C$ {j['preco']:.2f}")
                relatorio.append(f"    Finalizações/jogo: {j['finalizacoes_media']:.2f}")
                relatorio.append(f"    Gols/jogo: {j['gols_media']:.2f}")
                relatorio.append(f"    Taxa conversão: {j['taxa_conversao']:.1f}%")
                relatorio.append(f"    Média pontos: {j['pontos_media']:.2f}")
        
        # Laterais Ofensivos
        if deteccoes['laterais_ofensivos']:
            relatorio.append("\n\n⚡ LATERAIS OFENSIVOS")
            relatorio.append("-" * 70)
            for j in deteccoes['laterais_ofensivos']:
                relatorio.append(f"  {j['nome']} ({j['time']})")
                relatorio.append(f"    Preço: C$ {j['preco']:.2f}")
                relatorio.append(f"    Assistências/jogo: {j['assistencias_media']:.2f}")
                relatorio.append(f"    SG/jogo: {j['sg_media']:.2f}")
        
        # Zagueiros Artilheiros
        if deteccoes['zagueiros_artilheiros']:
            relatorio.append("\n\n⚽ ZAGUEIROS ARTILHEIROS")
            relatorio.append("-" * 70)
            for j in deteccoes['zagueiros_artilheiros']:
                relatorio.append(f"  {j['nome']} ({j['time']})")
                relatorio.append(f"    Preço: C$ {j['preco']:.2f}")
                relatorio.append(f"    Gols/jogo: {j['gols_media']:.2f}")
        
        # Volta Suspensão
        if deteccoes['volta_suspensao']:
            relatorio.append("\n\n🔄 VOLTANDO DE SUSPENSÃO")
            relatorio.append("-" * 70)
            for j in deteccoes['volta_suspensao']:
                relatorio.append(f"  {j['nome']} ({j['time']}) - {j['posicao']}")
                relatorio.append(f"    Preço: C$ {j['preco']:.2f} (BAIXO!)")
                relatorio.append(f"    Média antes: {j['media_antes_suspensao']:.2f}")
        
        relatorio.append("\n" + "="*70)
        
        return "\n".join(relatorio)
