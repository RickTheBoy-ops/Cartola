#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
========================================================================
CARTOLA FC - EXEMPLO DE USO
========================================================================
Script demonstrando uso do novo sistema refatorado.

Comparação com sistema antigo:
  ANTES: cartola_mega_optimizer(df, budget=100)
  AGORA: optimizer.optimize(df_enriched, budget=100)
========================================================================
"""

import sys
import os

# Adicionar src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from optimizer import CartolaOptimizer
from features import FeatureEngineeringV2


def main():
    """
    Exemplo completo de uso do sistema.
    """
    
    print("="*70)
    print("CARTOLA FC - SISTEMA DE OTIMIZAÇÃO V2")
    print("="*70)
    
    # ================================================================
    # 1. CARREGAR DADOS (simulado)
    # ================================================================
    
    print("\n📊 Carregando dados...")
    
    # Simular DataFrame de jogadores
    # Na prática, carregar do Cartola API ou banco de dados
    df_raw = pd.DataFrame({
        'atleta_id': range(1, 51),
        'nome': [f'Jogador_{i}' for i in range(1, 51)],
        'clube_id': [i % 20 + 1 for i in range(50)],
        'posicao_id': [(i % 5) + 1 for i in range(50)],
        'preco': [10 + (i % 30) for i in range(50)],
        'media': [5 + (i % 15) * 0.5 for i in range(50)],
        'pontos_ultimas_5': [4 + (i % 12) * 0.8 for i in range(50)],
        'jogos': [5 + (i % 10) for i in range(50)]
    })
    
    print(f"   ✅ {len(df_raw)} jogadores carregados")
    
    # ================================================================
    # 2. FEATURE ENGINEERING
    # ================================================================
    
    print("\n🧠 Executando Feature Engineering V2...")
    
    feature_eng = FeatureEngineeringV2()
    df_enriched = feature_eng.engineer_features(df_raw)
    
    # Visualizar top 10 por mega_score
    print("\n🎯 Top 10 Jogadores por Mega Score:")
    top_10 = feature_eng.get_top_players(df_enriched, n=10)
    
    for idx, row in top_10.iterrows():
        print(f"   {row['nome']:<15} | Pos: {row['posicao_id']} | "
              f"Mega: {row['mega_score']:.1f} | Preço: C${row['preco']:.1f}")
    
    # Estatísticas por posição
    print("\n📊 Estatísticas por Posição:")
    stats = feature_eng.get_position_stats(df_enriched)
    
    for pos_name, data in stats.items():
        print(f"   {pos_name}: {data['count']} jogadores | "
              f"Média: {data['mean_score']:.1f}")
    
    # ================================================================
    # 3. OTIMIZAÇÃO
    # ================================================================
    
    print("\n🚀 Iniciando otimização...")
    
    # Criar otimizador com estratégia 'mega'
    optimizer = CartolaOptimizer(
        strategy='mega',
        config={
            'max_players_per_club': 3,
            'enable_opponent_conflicts': True,
            'test_all_formations': True
        }
    )
    
    # Orçamento disponível
    budget = 100.0
    
    # Otimizar (testa todas as formações)
    lineup = optimizer.optimize(df_enriched, budget=budget)
    
    # ================================================================
    # 4. RESULTADOS
    # ================================================================
    
    if lineup is not None:
        print("\n" + "="*70)
        print("🏆 ESCALAÇÃO OTIMIZADA")
        print("="*70)
        
        # Validar escalação
        total_cost = lineup['preco'].sum()
        total_score = lineup['mega_score'].sum()
        
        print(f"\n💰 Custo Total: C$ {total_cost:.2f} / C$ {budget:.2f}")
        print(f"🎯 Score Total: {total_score:.2f}")
        print(f"⚽ Jogadores: {len(lineup)}")
        
        # Mostrar escalação
        print("\n👥 Lineup:")
        
        for idx, player in lineup.iterrows():
            print(f"   {player['nome']:<15} | Pos: {player['posicao_id']} | "
                  f"Mega: {player['mega_score']:>5.1f} | "
                  f"Média: {player['media']:>4.1f} | "
                  f"Preço: C${player['preco']:>5.1f}")
        
        # Selecionar capitão
        captain = optimizer.select_captain(lineup)
        print(f"\n🏺 Capitão: {captain}")
        
    else:
        print("\n❌ Nenhuma escalação viável encontrada!")
        print("   Dica: Aumente o orçamento ou revise as restrições.")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
