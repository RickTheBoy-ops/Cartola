"""Módulos de features avançadas para Cartola FC."""

from .valorizacao import SistemaValorizacao
from .confrontos import AnalisadorConfrontos
from .engineering import FeatureEngineer
from .detectors import DetectorPadroesEspeciais

__all__ = [
    'SistemaValorizacao',
    'AnalisadorConfrontos',
    'FeatureEngineer',
    'DetectorPadroesEspeciais'
]
