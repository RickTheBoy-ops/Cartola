"""
========================================================================
CARTOLA FC - OPTIMIZER PACKAGE
========================================================================
Sistema modular de otimização com múltiplas estratégias

Estratégias disponíveis:
  - MegaStrategy: Programação Linear (PuLP) + Feature Engineering V2
  - GeneticStrategy: Algoritmo Genético
  - EnsembleStrategy: Combina múltiplas abordagens

Uso:
    from src.optimizer import CartolaOptimizer
    
    optimizer = CartolaOptimizer(strategy='mega')
    lineup = optimizer.optimize(df, budget=100.0, formation='4-3-3')
========================================================================
"""

from .factory import CartolaOptimizer
from .base import OptimizerStrategy
from .mega_strategy import MegaStrategy

__all__ = ['CartolaOptimizer', 'OptimizerStrategy', 'MegaStrategy']
