#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ANÁLISE RÁPIDA - CARTOLA FC
Análise baseada puramente nos dados do Cartola (sem IA)
"""

import requests
import pandas as pd
from datetime import datetime

def load_cartola_data():
    """Carrega dados das APIs do Cartola"""
    
    API_ATLETAS = "https://api.cartola.globo.com/atletas/mercado"
    API_CLUBES = "https://api.cartola.globo.com/clubes"
    API_PARTIDAS = "https://api.cartola.globo.com/partidas"
    
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    print("📡 Carregando dados do Cartola FC...")
    print(f"⏰ Horário da consulta: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
    
    try:
        r1 = requests.get(API_ATLETAS, headers=HEADERS, timeout=30)
        r2 = requests.get(API_CLUBES, headers=HEADERS, timeout=30)
        r3 = requests.get(API_PARTIDAS, headers=HEADERS, timeout=30)
        
        print("✅ Dados carregados!\\n")
        return r1.json(), r2.json(), r3.json()
    
    except Exception as e:
        print(f"❌ Erro: {e}\\n")
        return None, None, None


def analise_completa():
    """Análise completa dos dados"""
    
    dados_atletas, dados_clubes, dados_partidas = load_cartola_data()
    
    if not dados_atletas:
        print("❌ Não foi possível carregar dados")
        return
    
    # Mapear clubes
    clubes_dict = {int(k): v.get('nome', 'Desconhecido') for k, v in dados_clubes.items()}
    
    # Processar atletas
    atletas = dados_atletas.get('atletas', [])
    if isinstance(atletas, dict):
        atletas = list(atletas.values())
    
    # Filtrar apenas jogadores provávies (status_id == 7)
    atletas_provaveis = [a for a in atletas if a.get('status_id') == 7]
    
    # Separar por posição
    POS_MAP = {1: "Goleiro", 2: "Lateral", 3: "Zagueiro", 4: "Meia", 5: "Atacante", 6: "Técnico"}
    
    print("=" * 100)
    print("🎯 ANÁLISE COMPLETA - CARTOLA FC")
    print("=" * 100)
    print()
    
    # ========== PARTIDAS ==========
    print("⚽ PRÓXIMOS CONFRONTOS:")
    print("-" * 100)
    
    partidas_lista = []
    if isinstance(dados_partidas, dict):
        for key, value in dados_partidas.items():
            if isinstance(value, dict) and 'partidas' in value:
                partidas_lista.extend(value['partidas'])
            elif isinstance(value, list):
                partidas_lista.extend(value)
            elif isinstance(value, dict) and 'clube_casa_id' in value:
                partidas_lista.append(value)
    
    for partida in partidas_lista[:15]:
        if not isinstance(partida, dict):
            continue
        casa_id = partida.get('clube_casa_id')
        visitante_id = partida.get('clube_visitante_id')
        
        if casa_id and visitante_id:
            casa_nome = clubes_dict.get(casa_id, 'Desconhecido')
            visitante_nome = clubes_dict.get(visitante_id, 'Desconhecido')
            print(f"   🏠 {casa_nome:25} vs  ✈️  {visitante_nome}")
    
    print()
    print()
    
    # ========== ANÁLISE POR POSIÇÃO ==========
    for pos_id in [1, 2, 3, 4, 5]:
        pos_nome = POS_MAP[pos_id]
        
        # Filtrar jogadores da posição
        pos_atletas = [a for a in atletas_provaveis if a.get('posicao_id') == pos_id]
        
        # Ordenar por média
        pos_atletas.sort(key=lambda x: x.get('media_num', 0), reverse=True)
        
        print(f"{'='*100}")
        print(f"📊 {pos_nome.upper()}")
        print(f"{'='*100}")
        print(f"{'Jogador':25} {'Clube':20} {'Média':>8} {'Preço':>10} {'Últ.':>8} {'Jogos':>6}")
        print("-" * 100)
        
        # Top 10 por média
        for i, atleta in enumerate(pos_atletas[:10], 1):
            nome = atleta.get('apelido', 'Desconhecido')[:24]
            clube = clubes_dict.get(atleta.get('clube_id'), '?')[:19]
            media = atleta.get('media_num', 0)
            preco = atleta.get('preco_num', 0)
            pontos = atleta.get('pontos_num', 0)
            jogos = atleta.get('jogos_num', 0)
            
            print(f"{i:2}. {nome:23} {clube:20} {media:8.2f} C${preco:8.2f} {pontos:8.2f} {jogos:6}")
        
        print()
        
        # BARATOS E BONS (custo-benefício)
        print(f"💰 CUSTO-BENEFÍCIO (Média > 5.0, Preço < C$10)")
        print("-" * 100)
        
        custo_beneficio = [a for a in pos_atletas if a.get('media_num', 0) > 5.0 and a.get('preco_num', 999) < 10]
        custo_beneficio.sort(key=lambda x: x.get('media_num', 0) / max(x.get('preco_num', 1), 0.1), reverse=True)
        
        if custo_beneficio:
            for i, atleta in enumerate(custo_beneficio[:5], 1):
                nome = atleta.get('apelido', 'Desconhecido')[:24]
                clube = clubes_dict.get(atleta.get('clube_id'), '?')[:19]
                media = atleta.get('media_num', 0)
                preco = atleta.get('preco_num', 0)
                ratio = media / max(preco, 0.1)
                
                print(f"{i}. {nome:23} {clube:20} Média: {media:5.2f} | Preço: C${preco:6.2f} | Ratio: {ratio:.3f}")
        else:
            print("   Nenhum jogador encontrado nesta faixa")
        
        print()
        print()
    
    # ========== CAPITÃES SUGERIDOS ==========
    print("=" * 100)
    print("👑 SUGESTÕES DE CAPITÃO (Top 5 por Média)")
    print("=" * 100)
    
    todos_provaveis = atletas_provaveis.copy()
    todos_provaveis.sort(key=lambda x: x.get('media_num', 0), reverse=True)
    
    print(f"{'Jogador':25} {'Posição':15} {'Clube':20} {'Média':>8} {'Preço':>10}")
    print("-" * 100)
    
    for i, atleta in enumerate(todos_provaveis[:5], 1):
        nome = atleta.get('apelido', 'Desconhecido')[:24]
        pos = POS_MAP.get(atleta.get('posicao_id'), '?')[:14]
        clube = clubes_dict.get(atleta.get('clube_id'), '?')[:19]
        media = atleta.get('media_num', 0)
        preco = atleta.get('preco_num', 0)
        
        print(f"{i}. {nome:23} {pos:15} {clube:20} {media:8.2f} C${preco:8.2f}")
    
    print()
    print("=" * 100)
    print()
    
    # ========== ESTATÍSTICAS GERAIS ==========
    print("📈 ESTATÍSTICAS GERAIS:")
    print("-" * 100)
    print(f"   Total de atletas disponíveis: {len(atletas)}")
    print(f"   Atletas prováveis (status 7): {len(atletas_provaveis)}")
    print(f"   Confrontos na rodada: {len(partidas_lista)}")
    
    # Preço médio por posição
    print()
    print("   💵 Preço médio por posição:")
    for pos_id in [1, 2, 3, 4, 5]:
        pos_atletas = [a for a in atletas_provaveis if a.get('posicao_id') == pos_id]
        if pos_atletas:
            preco_medio = sum(a.get('preco_num', 0) for a in pos_atletas) / len(pos_atletas)
            print(f"      {POS_MAP[pos_id]:12}: C$ {preco_medio:.2f}")
    
    print()
    print("=" * 100)
    print("✅ Análise concluída!")
    print("=" * 100)


if __name__ == "__main__":
    analise_completa()
