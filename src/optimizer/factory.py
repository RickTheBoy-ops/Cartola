#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
========================================================================
CARTOLA FC - OPTIMIZER FACTORY
========================================================================
Factory Pattern para criação de otimizadores.

Permite fácil adição de novas estratégias sem modificar código existente.
========================================================================
"""

from typing import Dict, Optional
import pandas as pd

from .base import OptimizerStrategy
from .mega_strategy import MegaStrategy


class CartolaOptimizer:
    """
    Factory para criação de otimizadores.
    
    Uso:
        optimizer = CartolaOptimizer(strategy='mega')
        lineup = optimizer.optimize(df, budget=100.0)
    """
    
    # Registro de estratégias disponíveis
    STRATEGIES = {
        'mega': MegaStrategy,
        # Futuras estratégias:
        # 'genetic': GeneticStrategy,
        # 'ensemble': EnsembleStrategy,
    }
    
    def __init__(self, strategy: str = 'mega', config: Optional[Dict] = None):
        """
        Args:
            strategy: Nome da estratégia ('mega', 'genetic', 'ensemble')
            config: Configurações específicas da estratégia
        """
        
        if strategy not in self.STRATEGIES:
            available = ', '.join(self.STRATEGIES.keys())
            raise ValueError(
                f"Estratégia '{strategy}' inválida. "
                f"Disponíveis: {available}"
            )
        
        # Instanciar estratégia
        strategy_class = self.STRATEGIES[strategy]
        self.strategy: OptimizerStrategy = strategy_class(config)
        
        print(f"🎯 Otimizador inicializado: {self.strategy.name}")
    
    def optimize(self, 
                 df: pd.DataFrame, 
                 budget: float, 
                 formation: Optional[str] = None,
                 **kwargs) -> Optional[pd.DataFrame]:
        """
        Otimiza escalação usando estratégia selecionada.
        
        Args:
            df: DataFrame com jogadores e features
            budget: Orçamento disponível
            formation: Formação específica ou None para testar todas
            **kwargs: Parâmetros adicionais
            
        Returns:
            DataFrame com escalação otimizada ou None
        """
        
        return self.strategy.optimize(df, budget, formation, **kwargs)
    
    def validate(self, lineup: pd.DataFrame, budget: float, formation: str) -> bool:
        """
        Valida escalação.
        
        Args:
            lineup: DataFrame com escalação
            budget: Orçamento máximo
            formation: Formação esperada
            
        Returns:
            True se válido
        """
        return self.strategy.validate(lineup, budget, formation)
    
    def select_captain(self, lineup: pd.DataFrame) -> str:
        """
        Seleciona capitão da escalação.
        
        Args:
            lineup: DataFrame com escalação
            
        Returns:
            Nome do capitão
        """
        return self.strategy.select_captain(lineup)
    
    def get_available_strategies(self) -> list:
        """
        Retorna lista de estratégias disponíveis.
        
        Returns:
            Lista de nomes de estratégias
        """
        return list(self.STRATEGIES.keys())
    
    def get_info(self) -> Dict:
        """
        Retorna informações sobre a estratégia atual.
        
        Returns:
            Dicionário com metadados
        """
        return self.strategy.get_info()
    
    @classmethod
    def register_strategy(cls, name: str, strategy_class: type):
        """
        Registra nova estratégia dinamicamente.
        
        Args:
            name: Nome da estratégia
            strategy_class: Classe que herda de OptimizerStrategy
        """
        
        if not issubclass(strategy_class, OptimizerStrategy):
            raise TypeError(
                f"{strategy_class} deve herdar de OptimizerStrategy"
            )
        
        cls.STRATEGIES[name] = strategy_class
        print(f"✅ Estratégia '{name}' registrada com sucesso!")
