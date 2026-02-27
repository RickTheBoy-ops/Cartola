#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
========================================================================
CARTOLA FC - AI ANALYZER (Perplexity)
========================================================================
Sistema de análise pura com IA Perplexity para Cartola FC:

🤖 Análise de jogadores e times
📊 Recomendações de escalação
⚽ Previsões de desempenho
🎯 Análise de confrontos
✅ Análise de lesões e notícias

API Key Perplexity: via variável de ambiente PERPLEXITY_API_KEY
========================================================================
"""

import requests
import json
import pandas as pd
from typing import Dict, List, Optional
import os

# ========================================================================
# CONFIGURAÇÃO
# ========================================================================

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "")

# ========================================================================
# CARREGAR DADOS DO CARTOLA
# ========================================================================

def load_cartola_data():
    """Carrega dados das APIs do Cartola"""
    
    API_ATLETAS = "https://api.cartola.globo.com/atletas/mercado"
    API_CLUBES = "https://api.cartola.globo.com/clubes"
    API_PARTIDAS = "https://api.cartola.globo.com/partidas"
    
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    print("📡 Carregando dados do Cartola FC...")
    
    try:
        r1 = requests.get(API_ATLETAS, headers=HEADERS, timeout=30)
        r2 = requests.get(API_CLUBES, headers=HEADERS, timeout=30)
        r3 = requests.get(API_PARTIDAS, headers=HEADERS, timeout=30)
        
        print("✅ Dados carregados!\n")
        return r1.json(), r2.json(), r3.json()
    
    except Exception as e:
        print(f"❌ Erro: {e}\n")
        return None, None, None


def parse_basic_data(dados_atletas, dados_clubes, dados_partidas):
    """Extrai informações básicas dos dados"""
    
    # Mapear clubes
    clubes_dict = {int(k): v.get('nome', 'Desconhecido') for k, v in dados_clubes.items()}
    
    # Mapear partidas
    partidas_lista = []
    if isinstance(dados_partidas, dict):
        for key, value in dados_partidas.items():
            if isinstance(value, dict) and 'partidas' in value:
                partidas_lista.extend(value['partidas'])
            elif isinstance(value, list):
                partidas_lista.extend(value)
            elif isinstance(value, dict) and 'clube_casa_id' in value:
                partidas_lista.append(value)
    
    # Extrair partidas
    partidas_info = []
    for partida in partidas_lista:
        if not isinstance(partida, dict):
            continue
        casa_id = partida.get('clube_casa_id')
        visitante_id = partida.get('clube_visitante_id')
        
        if casa_id and visitante_id:
            partidas_info.append({
                'casa': clubes_dict.get(casa_id, 'Desconhecido'),
                'visitante': clubes_dict.get(visitante_id, 'Desconhecido')
            })
    
    # Top jogadores por posição
    POS_MAP = {1: "Goleiro", 2: "Lateral", 3: "Zagueiro", 4: "Meia", 5: "Atacante", 6: "Técnico"}
    
    atletas = dados_atletas.get('atletas', [])
    if isinstance(atletas, dict):
        atletas = list(atletas.values())
    
    top_jogadores = {}
    for pos_id in [1, 2, 3, 4, 5]:
        pos_atletas = [a for a in atletas if a.get('posicao_id') == pos_id and a.get('status_id') == 7]
        pos_atletas.sort(key=lambda x: x.get('media_num', 0), reverse=True)
        
        top_5 = []
        for atleta in pos_atletas[:5]:
            top_5.append({
                'nome': atleta.get('apelido', 'Desconhecido'),
                'clube': clubes_dict.get(atleta.get('clube_id'), 'Desconhecido'),
                'preco': atleta.get('preco_num', 0),
                'media': atleta.get('media_num', 0),
                'ultima_pontuacao': atleta.get('pontos_num', 0)
            })
        
        top_jogadores[POS_MAP[pos_id]] = top_5
    
    return {
        'partidas': partidas_info,
        'top_jogadores': top_jogadores,
        'total_atletas': len(atletas)
    }


# ========================================================================
# PERPLEXITY AI - ANÁLISES
# ========================================================================

def perplexity_chat(prompt: str, system_prompt: str = None) -> str:
    """
    Faz uma consulta à Perplexity AI
    """
    
    url = "https://api.perplexity.ai/chat/completions"
    
    if not PERPLEXITY_API_KEY:
        return "⚠️ PERPLEXITY_API_KEY não configurada. Configure a variável de ambiente para habilitar a IA."
    
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }
    
    messages = []
    
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    messages.append({"role": "user", "content": prompt})
    
    payload = {
        "model": "llama-3.1-sonar-large-128k-online",
        "messages": messages,
        "max_tokens": 1000,
        "temperature": 0.4,
        "top_p": 0.9
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            return data['choices'][0]['message']['content']
        else:
            return f"❌ Erro {response.status_code}: {response.text}"
    
    except Exception as e:
        return f"❌ Erro na API: {str(e)}"


def analise_rodada_atual():
    """Analisa a rodada atual do Brasileirão"""
    
    print("\n" + "="*80)
    print("🤖 ANÁLISE DA RODADA - Perplexity AI")
    print("="*80 + "\n")
    
    system_prompt = """Você é um especialista em Cartola FC e Brasileirão.
Forneça análises objetivas, concisas e baseadas em dados recentes sobre:
- Forma dos times
- Lesões e suspensões
- Confrontos favoráveis
- Jogadores em destaque"""
    
    prompt = """Me dê uma análise completa da próxima rodada do Brasileirão 2024/2025 para o Cartola FC.

Inclua:
1. Quais times estão em melhor forma?
2. Principais jogadores para escalar (por posição: GOL, DEF, MEI, ATA)
3. Confrontos mais favoráveis para pontuação
4. Jogadores lesionados ou suspensos que devo evitar
5. Dicas de capitão para a rodada

Seja específico com nomes de jogadores e times."""
    
    print("⏳ Consultando IA... (pode levar 30-60 segundos)\n")
    
    resposta = perplexity_chat(prompt, system_prompt)
    
    print(resposta)
    print("\n" + "="*80 + "\n")
    
    return resposta


def analise_jogador_especifico(nome_jogador: str):
    """Analisa um jogador específico"""
    
    print("\n" + "="*80)
    print(f"🤖 ANÁLISE DO JOGADOR: {nome_jogador}")
    print("="*80 + "\n")
    
    system_prompt = """Você é um especialista em análise de desempenho de jogadores de futebol.
Forneça análises detalhadas baseadas em estatísticas recentes."""
    
    prompt = f"""Analise o jogador {nome_jogador} para o Cartola FC.

Inclua:
1. Forma recente (últimos 5 jogos)
2. Média de pontos no Cartola
3. Lesões ou problemas físicos
4. Próximos confrontos (favoráveis ou não)
5. Vale a pena escalar? Por quê?

Seja específico e objetivo."""
    
    print("⏳ Consultando IA...\n")
    
    resposta = perplexity_chat(prompt, system_prompt)
    
    print(resposta)
    print("\n" + "="*80 + "\n")
    
    return resposta


def analise_confronto(time_casa: str, time_visitante: str):
    """Analisa um confronto específico"""
    
    print("\n" + "="*80)
    print(f"🤖 ANÁLISE DO CONFRONTO: {time_casa} vs {time_visitante}")
    print("="*80 + "\n")
    
    system_prompt = """Você é um analista tático de futebol especializado em Cartola FC."""
    
    prompt = f"""Analise o confronto {time_casa} (casa) vs {time_visitante} (fora) para o Cartola FC.

Inclua:
1. Forma recente dos times
2. Histórico de confrontos
3. Jogadores que podem se destacar
4. Qual time tem vantagem defensiva/ofensiva?
5. Recomendações de quais jogadores escalar

Seja específico com nomes."""
    
    print("⏳ Consultando IA...\n")
    
    resposta = perplexity_chat(prompt, system_prompt)
    
    print(resposta)
    print("\n" + "="*80 + "\n")
    
    return resposta


def analise_com_dados_cartola(info_cartola: Dict):
    """Analisa os dados do Cartola FC com IA"""
    
    print("\n" + "="*80)
    print("🤖 ANÁLISE INTELIGENTE DOS DADOS DO CARTOLA")
    print("="*80 + "\n")
    
    # Preparar contexto
    contexto = "Dados do Cartola FC:\n\n"
    
    # Partidas
    if info_cartola['partidas']:
        contexto += "🏆 PRÓXIMOS CONFRONTOS:\n"
        for p in info_cartola['partidas'][:10]:
            contexto += f"  - {p['casa']} vs {p['visitante']}\n"
        contexto += "\n"
    
    # Top jogadores
    contexto += "⭐ TOP 5 JOGADORES POR POSIÇÃO (por média):\n\n"
    for pos, jogadores in info_cartola['top_jogadores'].items():
        if jogadores:
            contexto += f"{pos}:\n"
            for j in jogadores:
                contexto += f"  - {j['nome']} ({j['clube']}) - Média: {j['media']:.1f}, Preço: C${j['preco']:.1f}\n"
            contexto += "\n"
    
    system_prompt = """Você é um especialista em Cartola FC.
Analise os dados fornecidos e dê recomendações estratégicas."""
    
    prompt = f"""{contexto}

Com base nesses dados, me dê:

1. Top 3 jogadores IMPERDÍVEIS para escalar (considerando custo-benefício)
2. Confrontos mais favoráveis para pontuar
3. Posições onde vale investir mais dinheiro
4. Dica de capitão
5. Estratégia geral (focar em pontos ou valorização?)

Seja específico e justifique."""
    
    print("⏳ Consultando IA...\n")
    
    resposta = perplexity_chat(prompt, system_prompt)
    
    print(resposta)
    print("\n" + "="*80 + "\n")
    
    return resposta


# ========================================================================
# MENU INTERATIVO
# ========================================================================

def menu_principal():
    """Menu interativo para análises"""
    
    print("\n" + "="*80)
    print("🤖 CARTOLA FC - AI ANALYZER (Perplexity)")
    print("="*80)
    print("\nEscolha o tipo de análise:\n")
    print("1 - Análise da Rodada Atual (Panorama Geral)")
    print("2 - Análise de Jogador Específico")
    print("3 - Análise de Confronto Específico")
    print("4 - Análise Inteligente (baseada nos dados do Cartola)")
    print("5 - Análise Personalizada (pergunta livre)")
    print("0 - Sair")
    print("\n" + "="*80)
    
    escolha = input("\nEscolha uma opção: ").strip()
    
    if escolha == '1':
        analise_rodada_atual()
    
    elif escolha == '2':
        nome = input("\n⚽ Nome do jogador: ").strip()
        if nome:
            analise_jogador_especifico(nome)
    
    elif escolha == '3':
        casa = input("\n🏠 Time da casa: ").strip()
        visitante = input("✈️  Time visitante: ").strip()
        if casa and visitante:
            analise_confronto(casa, visitante)
    
    elif escolha == '4':
        print("\n📡 Carregando dados do Cartola FC...\n")
        dados_atletas, dados_clubes, dados_partidas = load_cartola_data()
        
        if dados_atletas:
            info = parse_basic_data(dados_atletas, dados_clubes, dados_partidas)
            analise_com_dados_cartola(info)
        else:
            print("❌ Não foi possível carregar dados do Cartola\n")
    
    elif escolha == '5':
        pergunta = input("\n❓ Sua pergunta sobre Cartola FC: ").strip()
        if pergunta:
            print("\n" + "="*80)
            print("🤖 ANÁLISE PERSONALIZADA")
            print("="*80 + "\n")
            
            system_prompt = "Você é um especialista em Cartola FC. Responda de forma objetiva e útil."
            
            print("⏳ Consultando IA...\n")
            resposta = perplexity_chat(pergunta, system_prompt)
            print(resposta)
            print("\n" + "="*80 + "\n")
    
    elif escolha == '0':
        print("\n👋 Até logo!\n")
        return False
    
    else:
        print("\n⚠️ Opção inválida!\n")
    
    return True


# ========================================================================
# MAIN
# ========================================================================

def main():
    """Loop principal"""
    
    continuar = True
    
    while continuar:
        continuar = menu_principal()
        
        if continuar:
            input("\n⏎ Pressione ENTER para voltar ao menu...")


if __name__ == "__main__":
    main()
