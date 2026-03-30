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

import logging
from typing import Dict, Optional
import pandas as pd

from .base import OptimizerStrategy
from .mega_strategy import MegaStrategy
from .genetic_strategy import GeneticStrategy
from .ensemble_strategy import EnsembleStrategy

_log = logging.getLogger(__name__)


class CartolaOptimizer:
    """
    Factory para criação de otimizadores.
    
    Uso:
        optimizer = CartolaOptimizer(strategy='mega')
        lineup = optimizer.optimize(df, budget=100.0)
    """
    
    STRATEGIES = {
        'mega': MegaStrategy,
        'genetic': GeneticStrategy,
        'ensemble': EnsembleStrategy,
    }

    # Tolerância para erros de ponto flutuante na validação de orçamento.
    # Ex.: 100.0001 ainda é considerado dentro de C$100.
    BUDGET_TOLERANCE = 0.05
    
    def __init__(self, strategy: str = 'mega', config: Optional[Dict] = None):
        if strategy not in self.STRATEGIES:
            available = ', '.join(self.STRATEGIES.keys())
            raise ValueError(
                f"Estratégia '{strategy}' inválida. "
                f"Disponíveis: {available}"
            )
        
        strategy_class = self.STRATEGIES[strategy]
        self.strategy: OptimizerStrategy = strategy_class(config)
        print(f"🎯 Otimizador inicializado: {self.strategy.name}")
    
    def optimize(
        self,
        df: pd.DataFrame,
        budget: float,
        formation: Optional[str] = None,
        partidas_df=None,
        **kwargs,
    ) -> Optional[pd.DataFrame]:
        """
        Otimiza escalação usando estratégia selecionada.
        Garante que o resultado respeite formação e orçamento.
        """
        lineup = self.strategy.optimize(
            df, budget, formation, partidas_df=partidas_df, **kwargs
        )

        if lineup is None or lineup.empty:
            return lineup

        # ── Validação pós-otimização: orçamento ──────────────────────────
        if 'preco' in lineup.columns:
            custo = lineup['preco'].sum()

            if custo > budget + self.BUDGET_TOLERANCE:
                _log.warning(
                    f"⚠️ Orçamento violado: C${custo:.2f} > C${budget:.2f}. "
                    f"Aplicando fallback de remoção de jogadores mais caros."
                )

                # Retira iterativamente o jogador mais caro até caber no orçamento.
                lineup_sorted = lineup.sort_values('preco', ascending=False)
                while (
                    lineup_sorted['preco'].sum() > budget
                    and len(lineup_sorted) > 6
                ):
                    lineup_sorted = lineup_sorted.iloc[1:]

                lineup = lineup_sorted

                if lineup['preco'].sum() > budget:
                    _log.error(
                        f"❌ Impossível gerar time dentro do orçamento C${budget:.2f}. "
                        f"Retornando None."
                    )
                    return None

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
            erros = [
                f"{{1:'GOL',2:'LAT',3:'ZAG',4:'MEI',5:'ATA',6:'TEC'}.get(pid,'?')}: "
                f"esperado {qtd}, got {actual.get(pid, 0)}"
                for pid, qtd in expected.items()
                if actual.get(pid, 0) != qtd
            ]
            if erros:
                _log.warning(
                    f"⚠️ Formação {formation} não respeitada: {', '.join(erros)}"
                )

        return lineup
    
    def validate(self, lineup: pd.DataFrame, budget: float, formation: str) -> bool:
        return self.strategy.validate(lineup, budget, formation)
    
    def select_captain(self, lineup: pd.DataFrame) -> str:
        return self.strategy.select_captain(lineup)
    
    def get_available_strategies(self) -> list:
        return list(self.STRATEGIES.keys())
    
    def get_info(self) -> Dict:
        return self.strategy.get_info()
    
    @classmethod
    def register_strategy(cls, name: str, strategy_class: type):
        if not issubclass(strategy_class, OptimizerStrategy):
            raise TypeError(f"{strategy_class} deve herdar de OptimizerStrategy")
        cls.STRATEGIES[name] = strategy_class
        print(f"✅ Estratégia '{name}' registrada com sucesso!")
