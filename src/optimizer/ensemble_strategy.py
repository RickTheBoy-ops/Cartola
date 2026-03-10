#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
========================================================================
CARTOLA FC - ENSEMBLE STRATEGY
========================================================================
Estratégia que combina e executa múltiplas estratégias baseadas
na Factory, retornando a melhor escalação dentre todas avaliadas.
========================================================================
"""

import pandas as pd
from typing import Dict, Optional, List
import logging

from .base import OptimizerStrategy

logger = logging.getLogger(__name__)

class EnsembleStrategy(OptimizerStrategy):
    """
    Estratégia de otimização Ensemble.
    
    Executa múltiplas estratégias e seleciona a escalação com maior score.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        
        # Configurações padrão
        self.default_config = {
            'strategies': ['mega', 'genetic'],  # Estratégias a serem executadas
            'strategy_configs': {}              # Configs específicas por estratégia
        }
        
        self.config = {**self.default_config, **(config or {})}
    
    def optimize(self, 
                 df: pd.DataFrame, 
                 budget: float, 
                 formation: Optional[str] = None,
                 **kwargs) -> Optional[pd.DataFrame]:
        """
        Otimização executando múltiplas estratégias e pegando a melhor.
        """
        # Para evitar imports circulares, importamos a factory aqui
        from .factory import CartolaOptimizer
        
        strategies_to_run = self.config.get('strategies', [])
        
        best_lineup = None
        best_score = -1
        best_strategy_name = None
        
        print(f"\n🔄 Iniciando ENSEMBLE STRATEGY ({len(strategies_to_run)} estratégias)...")
        
        for strategy_name in strategies_to_run:
            print(f"\n▶️ Executando {strategy_name.upper()}...")
            
            try:
                strat_config = self.config.get('strategy_configs', {}).get(strategy_name, {})
                optimizer = CartolaOptimizer(strategy=strategy_name, config=strat_config)
                
                lineup = optimizer.optimize(df, budget, formation, **kwargs)
                
                if lineup is not None and len(lineup) == 12:
                    score = self.calculate_score(lineup)
                    print(f"✅ {strategy_name.upper()} finalizado. Score: {score:.1f}")
                    
                    if score > best_score:
                        best_score = score
                        best_lineup = lineup
                        best_strategy_name = strategy_name
                else:
                    print(f"❌ {strategy_name.upper()} falhou em gerar line-up viável.")
            
            except Exception as e:
                print(f"⚠️ Erro ao executar {strategy_name}: {e}")
                logger.error(f"Erro no ensemble strategy '{strategy_name}': {e}")
                
        if best_lineup is not None:
             print(f"\n🏆 ENSEMBLE VENCEDOR: {best_strategy_name.upper()} (Score: {best_score:.1f})")
             
        return best_lineup
