# -*- coding: utf-8 -*-
"""
Módulo de otimização de escalações.
"""

from .factory import CartolaOptimizer
from .base import OptimizerStrategy
from .mega_strategy import MegaStrategy

__all__ = [
    'CartolaOptimizer',
    'OptimizerStrategy',
    'MegaStrategy'
]
