#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
========================================================================
CARTOLA FC - BASE OPTIMIZER STRATEGY
========================================================================
Classe abstrata base para todas as estratégias de otimização
Implementa Strategy Pattern para fácil adição de novas estratégias
========================================================================
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import pandas as pd


class OptimizerStrategy(ABC):
    """
    Classe abstrata base para estratégias de otimização.
    
    Todas as estratégias devem implementar os métodos:
    - optimize(): otimização principal
    - validate(): validação da solução
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Args:
            config: Dicionário de configurações específicas da estratégia
        """
        self.config = config or {}
        self.name = self.__class__.__name__
    
    @abstractmethod
    def optimize(self, 
                 df: pd.DataFrame, 
                 budget: float, 
                 formation: Optional[str] = None,
                 **kwargs) -> Optional[pd.DataFrame]:
        """
        Método principal de otimização.
        
        Args:
            df: DataFrame com jogadores e features
            budget: Orçamento disponível (cartoletas)
            formation: Formação tática (ex: '4-3-3') ou None para testar todas
            **kwargs: Parâmetros adicionais específicos da estratégia
            
        Returns:
            DataFrame com escalacao otimizada (12 jogadores) ou None se inválido
        """
        pass
    
    def validate(self, lineup: pd.DataFrame, budget: float, formation: str) -> bool:
        """
        Valida se a escalação é válida.
        
        Verifica:
        - 12 jogadores
        - Custo <= orçamento
        - Formação correta
        - Máximo 3 jogadores por clube
        
        Args:
            lineup: DataFrame com escalação
            budget: Orçamento máximo
            formation: Formação esperada
            
        Returns:
            True se válido, False caso contrário
        """
        
        if lineup is None or len(lineup) != 12:
            return False
        
        # Validar orçamento
        total_cost = lineup['preco'].sum()
        if total_cost > budget:
            return False
        
        # Validar formação
        if formation:
            expected_counts = self._parse_formation(formation)
            actual_counts = lineup.groupby('posicao_id').size().to_dict()
            
            for pos_id, expected in expected_counts.items():
                if actual_counts.get(pos_id, 0) != expected:
                    return False
        
        # Validar máximo por clube
        club_counts = lineup.groupby('clube_id').size()
        if (club_counts > 3).any():
            return False
        
        return True
    
    def _parse_formation(self, formation: str) -> Dict[int, int]:
        """
        Converte string de formação para dicionário {posicao_id: quantidade}.
        
        Args:
            formation: String formato '4-3-3'
            
        Returns:
            Dict mapeando posicao_id para quantidade esperada
        """
        
        formations = {
            '4-3-3': {1: 1, 2: 2, 3: 2, 4: 3, 5: 3, 6: 1},
            '3-4-3': {1: 1, 2: 0, 3: 3, 4: 4, 5: 3, 6: 1},
            '3-5-2': {1: 1, 2: 0, 3: 3, 4: 5, 5: 2, 6: 1},
            '4-4-2': {1: 1, 2: 2, 3: 2, 4: 4, 5: 2, 6: 1},
            '5-3-2': {1: 1, 2: 2, 3: 3, 4: 3, 5: 2, 6: 1},
            '5-4-1': {1: 1, 2: 2, 3: 3, 4: 4, 5: 1, 6: 1},
        }
        
        return formations.get(formation, formations['4-3-3'])
    
    def get_available_formations(self) -> List[str]:
        """
        Retorna lista de formações disponíveis.
        
        Returns:
            Lista de strings de formações
        """
        return ['4-3-3', '3-4-3', '3-5-2', '4-4-2', '5-3-2', '5-4-1']
    
    def select_captain(self, lineup: pd.DataFrame) -> str:
        """
        Seleciona o melhor capitão da escalação.
        
        Critérios:
        - Maior média
        - Bom momentum
        - Jogando em casa (bonus)
        
        Args:
            lineup: DataFrame com escalação
            
        Returns:
            Nome do capitão (apelido)
        """
        
        lineup_copy = lineup.copy()
        
        # Score de capitão
        lineup_copy['cap_score'] = (
            lineup_copy['media'] * 2.0
            + lineup_copy['ultima_pontuacao'] * 0.5
            + lineup_copy['is_home'].astype(float) * 1.0
            + lineup_copy.get('momentum', 1.0) * 0.5
        )
        
        cap_idx = lineup_copy['cap_score'].idxmax()
        return lineup_copy.loc[cap_idx, 'apelido']
    
    def calculate_score(self, lineup: pd.DataFrame) -> float:
        """
        Calcula score total da escalação.
        
        Args:
            lineup: DataFrame com escalação
            
        Returns:
            Score total (soma dos mega_scores individuais)
        """
        
        if 'mega_score' in lineup.columns:
            return lineup['mega_score'].sum()
        elif 'media' in lineup.columns:
            return lineup['media'].sum()
        else:
            return 0.0
    
    def get_info(self) -> Dict:
        """
        Retorna informações sobre a estratégia.
        
        Returns:
            Dicionário com metadados da estratégia
        """
        return {
            'name': self.name,
            'config': self.config,
            'formations': self.get_available_formations()
        }
