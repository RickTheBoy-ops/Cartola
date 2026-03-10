#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
========================================================================
CARTOLA FC - GENETIC STRATEGY
========================================================================
Estratégia de otimização usando Algoritmo Genético.

Adapta o GeneticTeamOptimizer (src.ml.optimizer) para a interface
padrão OptimizerStrategy.
========================================================================
"""

import pandas as pd
from typing import Dict, Optional
import logging

from .base import OptimizerStrategy
from src.ml.optimizer import GeneticTeamOptimizer

logger = logging.getLogger(__name__)

class GeneticStrategy(OptimizerStrategy):
    """
    Estratégia de otimização usando Algoritmo Genético.
    
    Aproveita a implementação do GeneticTeamOptimizer.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        
        # Configurações padrão
        self.default_config = {
            'population_size': 250,
            'generations': 150,
            'mutation_rate': 0.20,
            'elite_size': 20,
            'max_mesmo_clube': 3,
            'penalidade_variancia': True
        }
        
        self.config = {**self.default_config, **(config or {})}
    
    def optimize(self, 
                 df: pd.DataFrame, 
                 budget: float, 
                 formation: Optional[str] = None,
                 **kwargs) -> Optional[pd.DataFrame]:
        """
        Otimização principal usando Algoritmo Genético.
        
        Args:
            df: DataFrame com jogadores
            budget: Orçamento disponível
            formation: Formação tática (ex: '4-3-3'). Se None, usa '4-3-3' ou testa.
            
        Returns:
            DataFrame com melhor escalação ou None
        """
        if len(df) < 12:
            print(f"❌ Poucos jogadores disponíveis: {len(df)}")
            return None
            
        # O GeneticTeamOptimizer espera que as pontuações estejam na coluna 'predicao'
        # Se 'predicao' não existir, mas 'mega_score' existir, mapeamos
        pred_df = df.copy()
        if 'predicao' not in pred_df.columns:
            if 'mega_score' in pred_df.columns:
                pred_df['predicao'] = pred_df['mega_score']
            elif 'media' in pred_df.columns:
                pred_df['predicao'] = pred_df['media']
            else:
                pred_df['predicao'] = 0.0
                
        # Se formação não especificada, testaremos a padrão ou iteraremos (simplesmente '4-3-3' por default)
        formations_to_test = [formation] if formation else self.get_available_formations()
        
        best_lineup = None
        best_score = -1
        best_formation = None
        
        print(f"🧬 Iniciando otimização genética (pode demorar alguns segundos)...")
        
        for form in formations_to_test:
            try:
                optimizer = GeneticTeamOptimizer(
                    atletas_df=df,
                    predicoes=pred_df,
                    patrimonio=budget,
                    formacao=form,
                    population_size=self.config['population_size'],
                    generations=self.config['generations'],
                    mutation_rate=self.config['mutation_rate'],
                    elite_size=self.config['elite_size'],
                    max_mesmo_clube=self.config['max_mesmo_clube'],
                    penalidade_variancia=self.config['penalidade_variancia']
                )
                
                team, stats = optimizer.optimize()
                
                if team and len(team) == 12:
                    team_formatted = optimizer.format_team_output(team)
                    
                    # Restaurar os IDs para o formato do dataframe original
                    team_df = pd.DataFrame(team)
                    score = stats['total_pontos_preditos']
                    
                    if score > best_score:
                        best_score = score
                        best_lineup = team_df
                        best_formation = form
            except Exception as e:
                logger.error(f"Erro ao otimizar formação {form} com genético: {e}")
                
        if best_lineup is not None:
             print(f"\n   🏆 MELHOR FORMAÇÃO GENÉTICA: {best_formation} (Score: {best_score:.1f})")
             
        return best_lineup
