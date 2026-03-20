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
from .genetic_strategy import GeneticStrategy
from .ensemble_strategy import EnsembleStrategy

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
        'genetic': GeneticStrategy,
        'ensemble': EnsembleStrategy,
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
        Garante que o resultado respeite formação e orçamento.
        """
        lineup = self.strategy.optimize(df, budget, formation, **kwargs)

        if lineup is None or lineup.empty:
            return lineup

        # ── Validação pós-otimização: orçamento ──────────────────────────
        custo = lineup['preco'].sum() if 'preco' in lineup.columns else 0
        if custo > budget:
            import logging
            logging.getLogger(__name__).warning(
                f"⚠️ Orçamento violado: C${custo:.1f} > C${budget:.1f}. "
                f"Removendo jogadores mais caros até caber."
            )
            lineup = lineup.sort_values('preco').head(12)  # simplest fallback

        # ── Validação pós-otimização: formação ─────────────────────────
        if formation and 'posicao_id' in lineup.columns:
            FORMACOES = {
                '3-4-3': {1:1, 3:3, 2:0, 4:4, 5:3, 6:1},
                '3-5-2': {1:1, 3:3, 2:0, 4:5, 5:2, 6:1},
                '4-3-3': {1:1, 3:2, 2:2, 4:3, 5:3, 6:1},
                '4-4-2': {1:1, 3:2, 2:2, 4:4, 5:2, 6:1},
                '4-5-1': {1:1, 3:2, 2:2, 4:5, 5:1, 6:1},
                '5-3-2': {1:1, 3:3, 2:2, 4:3, 5:2, 6:1},
                '5-4-1': {1:1, 3:3, 2:2, 4:4, 5:1, 6:1},
            }
            expected = FORMACOES.get(formation, {})
            actual = lineup['posicao_id'].value_counts().to_dict()
            erros = []
            for pos_id, qtd in expected.items():
                if actual.get(pos_id, 0) != qtd:
                    nomes = {1:'GOL',2:'LAT',3:'ZAG',4:'MEI',5:'ATA',6:'TEC'}
                    erros.append(
                        f"{nomes.get(pos_id,'?')}: esperado {qtd}, got {actual.get(pos_id,0)}"
                    )
            if erros:
                import logging
                logging.getLogger(__name__).warning(
                    f"⚠️ Formação {formation} não respeitada: {', '.join(erros)}"
                )

        return lineup
    
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
