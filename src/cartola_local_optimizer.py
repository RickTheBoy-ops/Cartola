#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
========================================================================
CARTOLA FC - LOCAL OPTIMIZER (100% Offline)
========================================================================
Sistema COMPLETO rodando LOCALMENTE:

🤖 Ollama (Llama 3.2) - IA Local ao invés de Perplexity
🔍 DuckDuckGo Search - Scraping de notícias local
📰 Newspaper3k - Extração de texto de notícias
⚡ Zero custos de API
🔒 100% Privado e offline

Requisitos:
- pip install ollama duckduckgo-search newspaper3k trafilatura
- Ollama rodando: ollama serve (http://localhost:11434)
- Modelo baixado: ollama pull llama3.2
========================================================================
"""

import pandas as pd
import numpy as np
import requests
import json
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(__file__))

# ========================================================================
# IMPORTS LOCAIS
# ========================================================================

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("⚠️ Ollama não instalado. Instale com: pip install ollama")

try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False
    print("⚠️ DuckDuckGo Search não instalado. Instale com: pip install duckduckgo-search")

try:
    from newspaper import Article
    NEWSPAPER_AVAILABLE = True
except ImportError:
    NEWSPAPER_AVAILABLE = False
    print("⚠️ Newspaper3k não instalado. Instale com: pip install newspaper3k")

# ========================================================================
# CONFIGURAÇÕES
# ========================================================================

OLLAMA_MODEL = "llama3.2"  # Modelo padrão (pode usar llama3.1, mistral, etc)
OLLAMA_URL = "http://localhost:11434"

FORMATION_CONSTRAINTS = {
    '4-3-3': {1: 1, 2: 2, 3: 2, 4: 3, 5: 3, 6: 1},
    '3-4-3': {1: 1, 2: 0, 3: 3, 4: 4, 5: 3, 6: 1},
    '3-5-2': {1: 1, 2: 0, 3: 3, 4: 5, 5: 2, 6: 1},
    '4-4-2': {1: 1, 2: 2, 3: 2, 4: 4, 5: 2, 6: 1},
}

POS_MAP = {1: "GOL", 2: "LAT", 3: "ZAG", 4: "MEI", 5: "ATA", 6: "TEC"}

# ========================================================================
# FUNÇÕES DE ETL (Cartola FC APIs - Gratuitas)
# ========================================================================

def load_cartola_data():
    """Carrega dados das APIs GRATUITAS do Cartola FC"""
    
    API_ATLETAS = "https://api.cartola.globo.com/atletas/mercado"
    API_CLUBES = "https://api.cartola.globo.com/clubes"
    API_PARTIDAS = "https://api.cartola.globo.com/partidas"
    
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    print("📡 Carregando dados do Cartola FC (APIs Gratuitas)...")
    
    try:
        r1 = requests.get(API_ATLETAS, headers=HEADERS, timeout=30)
        r2 = requests.get(API_CLUBES, headers=HEADERS, timeout=30)
        r3 = requests.get(API_PARTIDAS, headers=HEADERS, timeout=30)
        
        print("   ✅ Dados carregados!")
        return r1.json(), r2.json(), r3.json()
    
    except Exception as e:
        print(f"   ❌ Erro: {e}")
        return None, None, None


def parse_data(dados_atletas, dados_clubes, dados_partidas):
    """Transforma dados brutos em DataFrame"""
    
    clubes_dict = {int(k): v.get('nome', 'Desconhecido') for k, v in dados_clubes.items()}
    
    partidas_dict = {}
    partidas_lista = []
    
    if isinstance(dados_partidas, dict):
        for key, value in dados_partidas.items():
            if isinstance(value, dict) and 'partidas' in value:
                partidas_lista.extend(value['partidas'])
            elif isinstance(value, list):
                partidas_lista.extend(value)
            elif isinstance(value, dict) and 'clube_casa_id' in value:
                partidas_lista.append(value)
    
    for partida in partidas_lista:
        if not isinstance(partida, dict):
            continue
        casa_id = partida.get('clube_casa_id')
        visitante_id = partida.get('clube_visitante_id')
        
        if casa_id and visitante_id:
            partidas_dict[casa_id] = {
                'opponent_id': visitante_id,
                'opponent_name': clubes_dict.get(visitante_id, 'Desconhecido'),
                'is_home_game': True
            }
            partidas_dict[visitante_id] = {
                'opponent_id': casa_id,
                'opponent_name': clubes_dict.get(casa_id, 'Desconhecido'),
                'is_home_game': False
            }
    
    atletas_lista = []
    atletas = dados_atletas.get('atletas', [])
    if isinstance(atletas, dict):
        atletas = list(atletas.values())
    
    for atleta in atletas:
        if atleta.get('status_id') != 7:
            continue
        
        clube_id = atleta.get('clube_id')
        info_partida = partidas_dict.get(clube_id, {})
        
        atletas_lista.append({
            'atleta_id': atleta.get('atleta_id'),
            'apelido': atleta.get('apelido', 'Desconhecido'),
            'posicao_id': atleta.get('posicao_id'),
            'posicao': POS_MAP.get(atleta.get('posicao_id'), 'Desconhecida'),
            'clube_id': clube_id,
            'clube': clubes_dict.get(clube_id, 'Desconhecido'),
            'preco': atleta.get('preco_num', 0.0),
            'media': atleta.get('media_num', 0.0),
            'variacao': atleta.get('variacao_num', 0.0),
            'ultima_pontuacao': atleta.get('pontos_num', 0.0),
            'jogos': atleta.get('jogos_num', 0),
            'minimo_valorizar': atleta.get('minimo_para_valorizar', 0.0),
            'opponent_id': info_partida.get('opponent_id'),
            'opponent_name': info_partida.get('opponent_name', 'Sem jogo'),
            'is_home_game': info_partida.get('is_home_game', False)
        })
    
    df = pd.DataFrame(atletas_lista)
    print(f"✅ {len(df)} atletas prováveis processados")
    
    return df


# ========================================================================
# LOCAL SETORISTA - Scraper de Notícias
# ========================================================================

def buscar_noticias_local(jogadores_top: List[Dict]) -> str:
    """
    Scraper LOCAL de notícias usando DuckDuckGo + Newspaper3k
    
    Args:
        jogadores_top: Lista dos top 5 jogadores da escalação
    
    Returns:
        String com contexto agregado de notícias
    """
    
    print("\n🔍 LOCAL SETORISTA - Buscando Notícias...")
    
    if not DDGS_AVAILABLE or not NEWSPAPER_AVAILABLE:
        print("   ⚠️ Bibliotecas de scraping não disponíveis")
        return "SEM_NOTICIAS"
    
    news_context = []
    
    try:
        ddgs = DDGS()
        
        for i, player in enumerate(jogadores_top[:5], 1):  # Top 5 jogadores
            nome = player['apelido']
            clube = player['clube']
            
            # Query de busca
            query = f"{nome} {clube} lesão dúvida escalação última hora"
            
            print(f"   [{i}/5] Buscando: {nome} ({clube})...")
            
            try:
                # Buscar no DuckDuckGo
                results = ddgs.text(query, max_results=3)
                
                for result in results[:2]:  # Top 2 resultados
                    url = result.get('href', result.get('link', ''))
                    
                    if not url:
                        continue
                    
                    try:
                        # Extrair texto com Newspaper3k
                        article = Article(url)
                        article.download()
                        article.parse()
                        
                        # Pegar primeiras 200 palavras
                        text = article.text[:500] if article.text else result.get('body', '')
                        
                        if text:
                            news_context.append(f"[{nome}] {text}")
                    
                    except Exception as e:
                        # Fallback: usar snippet da busca
                        snippet = result.get('body', '')
                        if snippet:
                            news_context.append(f"[{nome}] {snippet}")
            
            except Exception as e:
                print(f"      ⚠️ Erro na busca de {nome}: {e}")
                continue
        
        if len(news_context) > 0:
            print(f"   ✅ {len(news_context)} notícias encontradas!")
            return "\n\n".join(news_context)
        else:
            print("   ℹ️ Nenhuma notícia encontrada")
            return "SEM_NOTICIAS"
    
    except Exception as e:
        print(f"   ❌ Erro no scraper: {e}")
        return "SEM_NOTICIAS"


# ========================================================================
# VALIDAÇÃO COM OLLAMA (IA Local)
# ========================================================================

def validar_escalacao_com_ollama(lineup: pd.DataFrame, news_context: str) -> str:
    """
    Valida escalação usando Ollama (Llama 3.2 local)
    
    Args:
        lineup: DataFrame com a escalação
        news_context: Contexto de notícias do scraper
    
    Returns:
        Análise textual da IA
    """
    
    print("\n🤖 IA LOCAL (Ollama) - Validando Escalação...")
    
    if not OLLAMA_AVAILABLE:
        print("   ⚠️ Ollama não disponível. Análise baseada em stats.")
        return analise_baseada_em_stats(lineup)
    
    # Verificar se Ollama está rodando
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        if response.status_code != 200:
            print("   ⚠️ Ollama não está rodando. Inicie com: ollama serve")
            return analise_baseada_em_stats(lineup)
    except Exception:
        print("   ⚠️ Ollama não está rodando. Inicie com: ollama serve")
        return analise_baseada_em_stats(lineup)
    
    # Criar JSON da escalação
    escalacao_json = []
    
    for _, player in lineup.iterrows():
        escalacao_json.append({
            "posicao": player['posicao'],
            "jogador": player['apelido'],
            "clube": player['clube'],
            "adversario": player['opponent_name'],
            "casa": "Casa" if player['is_home_game'] else "Fora",
            "preco": float(player['preco']),
            "media": float(player['media']),
            "jogos": int(player['jogos'])
        })
    
    # System Prompt para Llama local
    system_prompt = """Você é um Analista de Fantasy Football especializado em Cartola FC.
    
Sua tarefa é analisar escalações e fornecer insights práticos.

FORMATO DE RESPOSTA:
---
NOTÍCIAS: [resumo de lesões/dúvidas baseado no contexto]
MATCHUPS: [análise dos confrontos]
NOTA: [0-10]/10
SUGESTÕES: [substituições se necessário]
---

Seja CONCISO (máximo 150 palavras)."""
    
    # Preparar contexto
    if news_context and news_context != "SEM_NOTICIAS":
        user_prompt = f"""Contexto de Notícias:
{news_context}

Escalação Proposta:
{json.dumps(escalacao_json, indent=2, ensure_ascii=False)}

Analise esta escalação considerando as notícias acima."""
    else:
        user_prompt = f"""Escalação Proposta:
{json.dumps(escalacao_json, indent=2, ensure_ascii=False)}

Analise esta escalação baseando-se nos dados estatísticos (média, jogos, confrontos)."""
    
    try:
        print(f"   🧠 Consultando {OLLAMA_MODEL}...")
        
        # Chamar Ollama
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            options={
                "temperature": 0.3,
                "top_p": 0.9,
                "num_predict": 300
            }
        )
        
        analise = response['message']['content']
        
        print("   ✅ Análise concluída!")
        
        return analise
    
    except Exception as e:
        print(f"   ❌ Erro no Ollama: {e}")
        print(f"   💡 Dica: Verifique se o modelo está baixado: ollama pull {OLLAMA_MODEL}")
        return analise_baseada_em_stats(lineup)


def analise_baseada_em_stats(lineup: pd.DataFrame) -> str:
    """Fallback: Análise baseada apenas em estatísticas"""
    
    avg_media = lineup['media'].mean()
    sem_historico = len(lineup[lineup['media'] == 0])
    jogadores_em_casa = len(lineup[lineup['is_home_game'] == True])
    
    nota = min(10, max(0, int(avg_media * 1.5)))
    
    analise = f"""
ANÁLISE AUTOMÁTICA (SEM IA):
---
NOTÍCIAS: Verificação manual recomendada
MATCHUPS: {jogadores_em_casa}/12 jogadores em casa
ESTATÍSTICAS: Média do time: {avg_media:.2f} | {sem_historico} sem histórico
NOTA: {nota}/10
SUGESTÕES: {'Escalação sólida' if nota >= 7 else 'Considere jogadores com mais histórico'}
---
⚠️ Ollama não disponível. Análise limitada a estatísticas básicas.
"""
    
    return analise


# ========================================================================
# FILTROS E OTIMIZAÇÃO
# ========================================================================

def filter_elite_players(df: pd.DataFrame, strategy: str):
    """Filtra apenas jogadores DE ELITE"""
    
    if strategy == 'POINTS':
        elite = (
            ((df['media'] >= 5.0) & (df['jogos'] >= 5)) |
            (df['preco'] >= 10.0)
        )
    else:
        elite = (
            ((df['media'] >= 3.0) & (df['jogos'] >= 3)) |
            ((df['preco'] >= 6.0) & (df['jogos'] > 0))
        )
    
    df_elite = df[elite].copy()
    
    print(f"✅ Filtro ELITE: {len(df_elite)} de {len(df)} jogadores")
    
    return df_elite


def check_opponent_conflict(selected_players: List, new_player: pd.Series) -> bool:
    """Verifica conflitos de adversários"""
    
    new_club = new_player['clube_id']
    new_opponent = new_player['opponent_id']
    new_pos = new_player['posicao_id']
    
    for player in selected_players:
        player_club = player['clube_id']
        player_opponent = player['opponent_id']
        player_pos = player['posicao_id']
        
        is_direct_opponent = (new_club == player_opponent) or (new_opponent == player_club)
        
        if is_direct_opponent:
            new_is_defender = new_pos in [1, 2, 3]
            player_is_defender = player_pos in [1, 2, 3]
            
            new_is_attacker = new_pos in [4, 5]
            player_is_attacker = player_pos in [4, 5]
            
            if (new_is_defender and player_is_attacker) or (new_is_attacker and player_is_defender):
                return True
    
    return False


def local_optimization(df: pd.DataFrame, budget: float, formation: str, strategy: str):
    """Otimização local completa"""
    
    print("\n🧮 Otimizando Escalação...")
    
    # Filtrar elite
    df = filter_elite_players(df, strategy)
    
    # Calcular score
    if strategy == 'POINTS':
        df['score'] = df['media'] * 3.0 + df['jogos'].clip(upper=10) * 0.5
    else:
        base = df['media'].fillna(3.0)
        mpv = df['minimo_valorizar'].where(df['minimo_valorizar'] != 0, base * 0.7)
        df['score'] = (base - mpv + 2).clip(lower=0) * 5 + (20.0 / (df['preco'] + 1.0)) * 3
    
    # Otimizar sem conflitos
    formation_req = FORMATION_CONSTRAINTS[formation]
    df_sorted = df.sort_values('score', ascending=False).copy()
    
    selected = []
    remaining_budget = budget
    pos_count = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    
    for _, player in df_sorted.iterrows():
        pos_id = player['posicao_id']
        price = player['preco']
        
        if pos_count[pos_id] >= formation_req[pos_id] or price > remaining_budget:
            continue
        
        if check_opponent_conflict(selected, player):
            continue
        
        selected.append(player)
        remaining_budget -= price
        pos_count[pos_id] += 1
        
        if sum(pos_count.values()) == 12:
            break
    
    lineup = pd.DataFrame(selected)
    
    print(f"✅ {len(lineup)} jogadores selecionados | C$ {lineup['preco'].sum():.2f}")
    
    return lineup


# ========================================================================
# DISPLAY
# ========================================================================

def display_local_lineup(lineup, analise_ia, budget):
    """Display da escalação com análise"""
    
    pos_order = {1: 1, 3: 2, 2: 3, 4: 4, 5: 5, 6: 6}
    lineup['pos_order'] = lineup['posicao_id'].map(pos_order)
    lineup = lineup.sort_values('pos_order')
    
    total_cost = lineup['preco'].sum()
    avg_media = lineup['media'].mean()
    
    capitao_idx = lineup['media'].idxmax() if lineup['media'].max() > 0 else lineup.iloc[0].name
    capitao = lineup.loc[capitao_idx]
    
    print("\n" + "="*80)
    print("🏆 ESCALAÇÃO LOCAL (100% Offline)")
    print("="*80 + "\n")
    
    display_df = lineup[['posicao', 'apelido', 'clube', 'opponent_name', 'is_home_game', 'preco', 'media']].copy()
    display_df.columns = ['Pos', 'Jogador', 'Clube', 'Adversário', 'Casa?', 'Preço', 'Média']
    display_df['Casa?'] = display_df['Casa?'].apply(lambda x: '🏠' if x else '✈️')
    display_df['Preço'] = display_df['Preço'].apply(lambda x: f"C${x:.1f}")
    
    display_df['Jogador'] = display_df.apply(
        lambda row: f"⭐{row['Jogador']}" if row['Jogador'] == capitao['apelido'] else row['Jogador'],
        axis=1
    )
    
    print(display_df.to_string(index=False))
    
    print("\n" + "-"*80)
    print(f"💰 CUSTO: C$ {total_cost:.2f} / {budget:.2f}")
    print(f"📊 MÉDIA: {avg_media:.2f}")
    print(f"⭐ CAPITÃO: {capitao['apelido']}")
    print("-"*80)
    
    # Análise da IA LOCAL
    print("\n" + "="*80)
    print("🤖 ANÁLISE DA IA LOCAL (Ollama)")
    print("="*80)
    print(analise_ia)
    print("="*80)
    
    return lineup


# ========================================================================
# MAIN
# ========================================================================

def main():
    """Pipeline LOCAL completo"""
    
    print("\n" + "="*80)
    print("🤖 CARTOLA FC - LOCAL OPTIMIZER (100% Offline)")
    print("="*80)
    print("✅ Ollama (Llama 3.2) - IA rodando localmente")
    print("✅ DuckDuckGo Search - Scraping de notícias")
    print("✅ Zero custos de API")
    print("="*80 + "\n")
    
    # Verificar dependências
    if not OLLAMA_AVAILABLE:
        print("⚠️ AVISO: Ollama não instalado")
        print("   Instale com: pip install ollama")
        print("   Baixe o modelo: ollama pull llama3.2\n")
    
    # Input
    budget = float(input("💰 Orçamento: "))
    
    print("\n📊 Estratégia:")
    print("   1 - PONTOS")
    print("   2 - VALORIZAÇÃO")
    strategy_choice = input("Escolha (1 ou 2): ")
    strategy = 'POINTS' if strategy_choice == '1' else 'WEALTH'
    
    print("\n⚽ Formação:")
    print("   1 - 4-3-3  |  2 - 3-4-3  |  3 - 3-5-2  |  4 - 4-4-2")
    formation_choice = input("Escolha (1-4): ")
    formations = ['4-3-3', '3-4-3', '3-5-2', '4-4-2']
    formation = formations[int(formation_choice) - 1]
    
    print("\n" + "="*80)
    
    # Pipeline
    dados_atletas, dados_clubes, dados_partidas = load_cartola_data()
    
    if not dados_atletas:
        print("\n❌ Erro ao carregar dados!")
        return
    
    df = parse_data(dados_atletas, dados_clubes, dados_partidas)
    
    # Otimizar
    lineup = local_optimization(df, budget, formation, strategy)
    
    if len(lineup) < 12:
        print("\n❌ Não foi possível gerar escalação completa!")
        return
    
    # Buscar notícias (top 5 por média)
    lineup_sorted = lineup.sort_values('media', ascending=False)
    jogadores_top = lineup_sorted.head(5).to_dict('records')
    
    news_context = buscar_noticias_local(jogadores_top)
    
    # Validar com IA local
    analise_ia = validar_escalacao_com_ollama(lineup, news_context)
    
    # Display
    final_lineup = display_local_lineup(lineup, analise_ia, budget)
    
    # Salvar
    output_file = f'local_lineup_{strategy}_{formation.replace("-", "")}.xlsx'
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        final_lineup.to_excel(writer, sheet_name='Escalação Local', index=False)
        
        df_ia = pd.DataFrame({'Análise IA Local': [analise_ia]})
        df_ia.to_excel(writer, sheet_name='Análise Ollama', index=False)
    
    print(f"\n✅ Escalação salva em: {output_file}")
    print("\n" + "="*80)
    print("🏆 OTIMIZAÇÃO LOCAL CONCLUÍDA!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
