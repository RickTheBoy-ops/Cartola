#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
========================================================================
CARTOLA FC - MEGA STRATEGY (PuLP Optimization)
========================================================================
Estratégia de otimização usando Programação Linear Inteira (PuLP).

Refatorado de: cartola_mega_optimizer.py

Características:
  - Programação Linear para solução matematicamente ótima
  - Feature Engineering V2 (análise por posição)
  - Restrições: formação, orçamento, conflitos adversários
  - Testa todas as formações e escolhe a melhor
========================================================================
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')

try:
    from pulp import (LpProblem, LpMaximize, LpVariable, lpSum, 
                      LpStatus, PULP_CBC_CMD, value)
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False
    print("⚠️ PuLP não instalado. Instale com: pip install pulp")

from .base import OptimizerStrategy


class MegaStrategy(OptimizerStrategy):
    """
    Estratégia de otimização usando PuLP (Programação Linear).
    
    Objetivo: Maximizar mega_score total respeitando restrições.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        
        if not PULP_AVAILABLE:
            raise ImportError(
                "PuLP não disponível! Instale com: pip install pulp"
            )
        
        # Configurações padrão
        self.default_config = {
            'max_players_per_club': 3,
            'enable_opponent_conflicts': True,
            'solver_time_limit': 30,
            'test_all_formations': True
        }
        
        self.config = {**self.default_config, **(config or {})}
    
    def optimize(self, 
                 df: pd.DataFrame, 
                 budget: float, 
                 formation: Optional[str] = None,
                 **kwargs) -> Optional[pd.DataFrame]:
        """
        Otimização principal usando PuLP.
        
        Args:
            df: DataFrame com jogadores e features (deve ter 'mega_score')
            budget: Orçamento disponível
            formation: Formação específica ou None para testar todas
            
        Returns:
            DataFrame com melhor escalação ou None
        """
        
        if len(df) < 12:
            print(f"❌ Poucos jogadores disponíveis: {len(df)}")
            return None
        
        # Se formação específica, otimizar apenas ela
        if formation:
            return self._optimize_single_formation(df, budget, formation)
        
        # Caso contrário, testar todas
        return self._optimize_all_formations(df, budget)
    
    def _optimize_all_formations(self, 
                                 df: pd.DataFrame, 
                                 budget: float) -> Optional[pd.DataFrame]:
        """
        Testa todas as formações e retorna a melhor.
        """
        
        formations = self.get_available_formations()
        
        best_lineup = None
        best_formation = None
        best_score = -1
        
        print(f"\n🧠 Testando {len(formations)} formações...")
        
        for form in formations:
            lineup = self._optimize_single_formation(df, budget, form)
            
            if lineup is not None and len(lineup) == 12:
                score = self.calculate_score(lineup)
                cost = lineup['preco'].sum()
                
                print(f"   ⚽ {form}: Score={score:.1f} | Custo=C${cost:.1f}")
                
                if score > best_score:
                    best_score = score
                    best_lineup = lineup
                    best_formation = form
            else:
                print(f"   ❌ {form}: Inviável com este orçamento")
        
        if best_lineup is not None:
            print(f"\n   🏆 MELHOR FORMAÇÃO: {best_formation} (Score: {best_score:.1f})")
        else:
            print("\n   ❌ Nenhuma formação viável encontrada!")
        
        return best_lineup
    
    def _optimize_single_formation(self, 
                                   df: pd.DataFrame, 
                                   budget: float, 
                                   formation: str) -> Optional[pd.DataFrame]:
        """
        Otimiza para uma formação específica usando PuLP.
        """
        
        formation_req = self._parse_formation(formation)
        
        # Criar problema de otimização
        prob = LpProblem(f"CartolaMega_{formation}", LpMaximize)
        
        # Variáveis de decisão (0 ou 1 para cada jogador)
        player_vars = {}
        for idx in df.index:
            player_vars[idx] = LpVariable(f"x_{idx}", cat='Binary')
        
        # OBJETIVO: Maximizar mega_score total
        prob += lpSum(
            df.loc[idx, 'mega_score'] * player_vars[idx] 
            for idx in df.index
        )
        
        # RESTRIÇÃO 1: Quantidade por posição (formação)
        for pos_id, count in formation_req.items():
            pos_indices = df[df['posicao_id'] == pos_id].index
            prob += lpSum(player_vars[idx] for idx in pos_indices) == count
        
        # RESTRIÇÃO 2: Total de 12 jogadores
        prob += lpSum(player_vars[idx] for idx in df.index) == 12
        
        # RESTRIÇÃO 3: Orçamento
        prob += lpSum(
            df.loc[idx, 'preco'] * player_vars[idx] 
            for idx in df.index
        ) <= budget
        
        # RESTRIÇÃO 4: Máximo de jogadores por clube
        max_per_club = self.config['max_players_per_club']
        for clube_id in df['clube_id'].unique():
            clube_indices = df[df['clube_id'] == clube_id].index
            prob += lpSum(
                player_vars[idx] for idx in clube_indices
            ) <= max_per_club
        
        # RESTRIÇÃO 5: Conflitos de adversários (DEF vs ATK)
        if self.config['enable_opponent_conflicts']:
            conflicts = self._get_conflict_pairs(df)
            for idx_i, idx_j in conflicts:
                prob += player_vars[idx_i] + player_vars[idx_j] <= 1
        
        # Resolver
        solver = PULP_CBC_CMD(
            msg=0, 
            timeLimit=self.config['solver_time_limit']
        )
        prob.solve(solver)
        
        # Extrair solução
        if LpStatus[prob.status] != 'Optimal':
            return None
        
        selected_indices = [
            idx for idx in df.index 
            if value(player_vars[idx]) == 1
        ]
        
        lineup = df.loc[selected_indices].copy()
        
        return lineup
    
    def _get_conflict_pairs(self, df: pd.DataFrame) -> List[Tuple[int, int]]:
        """
        Identifica pares de jogadores com conflito de adversários.
        
        Regra: NÃO escalar defensor + atacante/meia de times adversários diretos.
        (Ex: Zagueiro PAL + Atacante FLU quando PAL x FLU)
        
        Args:
            df: DataFrame com jogadores
            
        Returns:
            Lista de tuplas (idx_i, idx_j) com conflitos
        """
        
        conflicts = []
        indices = df.index.tolist()
        
        for i, idx_i in enumerate(indices):
            pi = df.loc[idx_i]
            
            for idx_j in indices[i+1:]:
                pj = df.loc[idx_j]
                
                # Verificar se são adversários diretos
                is_opponents = (
                    (pi['clube_id'] == pj.get('opponent_id')) or
                    (pi.get('opponent_id') == pj['clube_id'])
                )
                
                if not is_opponents:
                    continue
                
                # Conflito: defensor vs atacante/meia
                pi_def = pi['posicao_id'] in [1, 2, 3]
                pj_def = pj['posicao_id'] in [1, 2, 3]
                pi_atk = pi['posicao_id'] in [4, 5]
                pj_atk = pj['posicao_id'] in [4, 5]
                
                if (pi_def and pj_atk) or (pi_atk and pj_def):
                    conflicts.append((idx_i, idx_j))
        
        return conflicts
