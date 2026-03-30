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


# FIX: Probabilidade de troca por posição (antes era 0.45 fixo para todas)
# GOL é mais estável → menor prob; ATA tem alta variância → maior prob
_PROB_TROCA_POR_POSICAO = {
    1: 0.25,  # GOL
    2: 0.35,  # LAT
    3: 0.35,  # ZAG
    4: 0.45,  # MEI
    5: 0.55,  # ATA
}


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
                 partidas_df=None,
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
            
        # ── Regra Anti-Confronto ──────────
        self.confrontos_set: set = set()
        if partidas_df is not None and len(partidas_df) > 0:
            for _, row in partidas_df.iterrows():
                c_a = row.get('clube_casa_id') or row.get('clube_id_a')
                c_b = row.get('clube_visitante_id') or row.get('clube_id_b')
                if c_a and c_b and c_a != c_b:
                    self.confrontos_set.add(frozenset({int(c_a), int(c_b)}))
        
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
        
        # RESTRIÇÃO 5: Conflitos de adversários (ZAG/LAT vs ATA/MEI)
        # FIX: GOL removido do grupo de defensores no conflito.
        # Goleiro não conflita com atacante adversário — pelo contrário,
        # um GOL em jogo difícil pode ter boa pontuação com defesas e pegadas.
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

        Regra corrigida: NÃO escalar ZAG/LAT + ATA/MEI de times adversários diretos.
        GOL foi removido do grupo de defensores — goleiro não conflita com atacante
        adversário (pode pontuar bem em jogo difícil com defesas e pegadas).

        FIX v1: antes usava posicao_id [1,2,3] como defensores, incluindo GOL.
                Agora usa apenas [2,3] (LAT e ZAG) como defensores no conflito.
        
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
                
                # Verificar se são adversários diretos usando o confrontos_set
                c_i = pi.get('clube_id', 0)
                c_j = pj.get('clube_id', 0)
                is_opponents = False
                if c_i and c_j and c_i != c_j:
                    if hasattr(self, 'confrontos_set') and frozenset({int(c_i), int(c_j)}) in self.confrontos_set:
                        is_opponents = True
                
                if not is_opponents:
                    continue
                
                # FIX: GOL (pos 1) removido do grupo de defensores.
                # Apenas LAT (2) e ZAG (3) conflitam com ATA/MEI adversários.
                pi_def = pi['posicao_id'] in [2, 3]  # era [1, 2, 3]
                pj_def = pj['posicao_id'] in [2, 3]  # era [1, 2, 3]
                pi_atk = pi['posicao_id'] in [4, 5]
                pj_atk = pj['posicao_id'] in [4, 5]
                
                if (pi_def and pj_atk) or (pi_atk and pj_def):
                    conflicts.append((idx_i, idx_j))
        
        return conflicts

    # ---------------------------------------------------------------
    # RESERVA DE LUXO (regra 2025/2026)
    # ---------------------------------------------------------------

    def optimize_with_luxury_reserve(
        self,
        df: pd.DataFrame,
        budget: float,
        formation: Optional[str] = None,
    ) -> Optional[Dict]:
        """
        Otimização que modela a regra de Reserva de Luxo do Cartola FC.

        Regra Reserva de Luxo (2025/2026):
            - Você escolhe UM reserva por posição (goleiro, lateral, zagueiro,
              meia, atacante).
            - Se o titular da mesma posição pontuar MENOS que o reserva,
              o sistema troca automaticamente.
            - Isso aumenta o EV (valor esperado) da escalação porque você
              captura o upside de duas opções por posição.

        Modelagem no solver:
            - Para cada posição, selecionamos 1 titular + 1 reserva potencial.
            - Adicionamos uma variável auxiliar 'y_p' que representa o ganho
              esperado da troca (E[max(titular, reserva)] - E[titular]).
            - A função objetivo maximiza a predição dos titulares +
              (fator_reserva * ganho esperado de cada reserva).
            - FIX: budget split agora é dinâmico — os titulares usam o orçamento
              ótimo real, e as reservas ficam com o restante (antes era 80% fixo).

        Args:
            df:        DataFrame com atletas, 'mega_score', 'preco', 'posicao_id'
            budget:    Orçamento total (titulares + reservas)
            formation: Formação dos titulares (ex: '4-3-3'). None = testa todas.

        Returns:
            Dict com:
                'titulares':  DataFrame com 12 titulares (11+técnico)
                'reservas':   DataFrame com 5 reservas de luxo (1 por posição exceto TEC)
                'score_total': score previsto com bônus de reserva
                'custo_total': custo total dos 17 jogadores
        """
        if not PULP_AVAILABLE:
            raise ImportError("PuLP não disponível! Instale com: pip install pulp")

        if 'mega_score' not in df.columns:
            raise ValueError("DataFrame precisa da coluna 'mega_score'")

        # Mapeia posições (1=GOL, 2=LAT, 3=ZAG, 4=MEI, 5=ATA, 6=TEC)
        POSICOES_COM_RESERVA = [1, 2, 3, 4, 5]

        # ---- Step 1: encontrar titulares ----
        # FIX: budget dinâmico — tenta alocar o máximo possível para os titulares,
        # reservando um mínimo razoável para as reservas (5 jogadores baratos).
        # Antes era fixo em 80%, o que prejudicava orçamentos apertados.
        RESERVA_MINIMA = 5 * 4.0   # estimativa: 5 reservas a C$4 cada
        budget_titulares = max(budget - RESERVA_MINIMA, budget * 0.80)
        titulares = self.optimize(df, budget_titulares, formation)
        if titulares is None:
            # Fallback: tenta com orçamento total se o dinâmico falhar
            titulares = self.optimize(df, budget, formation)
        if titulares is None:
            return None

        titulares_idx = set(titulares.index)
        orcamento_reservas = budget - titulares['preco'].sum()

        if orcamento_reservas <= 0:
            return {
                'titulares': titulares,
                'reservas': pd.DataFrame(),
                'score_total': self.calculate_score(titulares),
                'custo_total': titulares['preco'].sum(),
            }

        # ---- Step 2: selecionar reservas de luxo por posição ----
        reservas_selecionadas = []
        orcamento_restante = orcamento_reservas

        for pos_id in POSICOES_COM_RESERVA:
            # Titulares dessa posição
            titulares_pos = titulares[titulares['posicao_id'] == pos_id]
            if titulares_pos.empty:
                continue

            # Preço médio dos titulares dessa posição
            preco_medio_titular = titulares_pos['preco'].mean()

            # Candidatos a reserva: mesma posição, não titular, dentro do orçamento restante
            candidatos = df[
                (df['posicao_id'] == pos_id) &
                (~df.index.isin(titulares_idx)) &
                (df['preco'] <= orcamento_restante) &
                (df['preco'] <= preco_medio_titular * 1.2)
            ].copy()

            if candidatos.empty:
                continue

            # FIX: probabilidade de troca agora é específica por posição.
            # GOL é mais estável (0.25), ATA tem alta variância (0.55).
            # Antes era 0.45 fixo para qualquer posição, distorcendo o ranking de reservas.
            prob_troca = _PROB_TROCA_POR_POSICAO.get(pos_id, 0.40)
            score_titular_medio = titulares_pos['mega_score'].mean()

            candidatos['ev_reserva'] = np.where(
                candidatos['mega_score'] > score_titular_medio,
                (candidatos['mega_score'] - score_titular_medio) * prob_troca,
                candidatos['mega_score'] * 0.10  # piso: seguro mínimo
            )

            # Selecionar o melhor EV dentro do orçamento
            melhor_reserva = candidatos.nlargest(1, 'ev_reserva').iloc[0]
            reservas_selecionadas.append(melhor_reserva)
            orcamento_restante -= melhor_reserva['preco']
            titulares_idx.add(melhor_reserva.name)

            # Se acabou o orçamento, para
            if orcamento_restante <= 0:
                break

        reservas_df = pd.DataFrame(reservas_selecionadas) if reservas_selecionadas else pd.DataFrame()

        # ---- Step 3: calcular score total com bônus de reserva ----
        score_titulares = self.calculate_score(titulares)
        bonus_reserva = reservas_df['ev_reserva'].sum() if not reservas_df.empty else 0
        score_total = score_titulares + bonus_reserva

        custo_total = (
            titulares['preco'].sum() +
            (reservas_df['preco'].sum() if not reservas_df.empty else 0)
        )

        return {
            'titulares': titulares,
            'reservas': reservas_df,
            'score_total': score_total,
            'score_titulares': score_titulares,
            'bonus_reserva': bonus_reserva,
            'custo_total': custo_total,
        }

    def print_lineup_with_reserve(self, result: Dict) -> None:
        """
        Imprime o time completo (titulares + reservas de luxo) de forma legível.
        """
        if result is None:
            print("❌ Nenhuma escalação gerada.")
            return

        titulares = result['titulares']
        reservas = result.get('reservas', pd.DataFrame())

        POS_NOME = {1: 'GOL', 2: 'LAT', 3: 'ZAG', 4: 'MEI', 5: 'ATA', 6: 'TEC'}

        print("\n" + "=" * 60)
        print("🏆 ESCALAÇÃO ÓTIMA COM RESERVA DE LUXO")
        print("=" * 60)

        print(f"\n⚽ TITULARES ({len(titulares)} jogadores)")
        print("-" * 60)
        for _, row in titulares.sort_values('posicao_id').iterrows():
            pos = POS_NOME.get(int(row.get('posicao_id', 0)), '?')
            nome = row.get('apelido', row.get('nome', 'N/A'))
            score = row.get('mega_score', 0)
            preco = row.get('preco', 0)
            selo = row.get('selo_valorizacao', '')
            print(f"  [{pos}] {nome:<20} Score={score:>6.1f}  C${preco:>5.1f}  {selo}")

        if not reservas.empty:
            print(f"\n🔄 RESERVAS DE LUXO ({len(reservas)} jogadores)")
            print("-" * 60)
            for _, row in reservas.sort_values('posicao_id').iterrows():
                pos = POS_NOME.get(int(row.get('posicao_id', 0)), '?')
                nome = row.get('apelido', row.get('nome', 'N/A'))
                ev = row.get('ev_reserva', 0)
                preco = row.get('preco', 0)
                print(f"  [{pos}] {nome:<20} EV_reserva={ev:>5.1f}  C${preco:>5.1f}")

        print(f"\n{'=' * 60}")
        print(f"💰 Custo total:     C${result['custo_total']:.1f}")
        print(f"🎯 Score titulares: {result.get('score_titulares', 0):.1f}")
        print(f"🔄 Bônus reserva:   +{result.get('bonus_reserva', 0):.1f}")
        print(f"⭐ Score total EV:  {result['score_total']:.1f}")
        print("=" * 60)
