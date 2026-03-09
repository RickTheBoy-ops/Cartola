#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
========================================================================
CARTOLA FC - ULTIMATE OPTIMIZER (AI-POWERED)
========================================================================
Sistema COMPLETO com IA integrada no CORE:

🤖 IA PERPLEXITY valida TODA escalação automaticamente
📡 RAPIDAPI busca forma dos times em tempo real
⚡ Multipliers automáticos baseados em forma (WWWWW = 1.1x)
🎯 Análise de riscos (lesões, poupados) via IA
✅ Tudo roda AUTOMATICAMENTE - Zero interação manual

Credenciais:
- RapidAPI: via variável de ambiente RAPIDAPI_KEY
- Perplexity: via variável de ambiente PERPLEXITY_API_KEY
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
# CONFIGURAÇÕES E FUNÇÕES BASE
# ========================================================================

FORMATION_CONSTRAINTS = {
    '4-3-3': {1: 1, 2: 2, 3: 2, 4: 3, 5: 3, 6: 1},
    '3-4-3': {1: 1, 2: 0, 3: 3, 4: 4, 5: 3, 6: 1},
    '3-5-2': {1: 1, 2: 0, 3: 3, 4: 5, 5: 2, 6: 1},
    '4-4-2': {1: 1, 2: 2, 3: 2, 4: 4, 5: 2, 6: 1},
}

POS_MAP = {1: "GOL", 2: "LAT", 3: "ZAG", 4: "MEI", 5: "ATA", 6: "TEC"}


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
        
        print("   ✅ Dados carregados!")
        return r1.json(), r2.json(), r3.json()
    
    except Exception as e:
        print(f"   ❌ Erro: {e}")
        return None, None, None


def parse_data(dados_atletas, dados_clubes, dados_partidas):
    """Transforma dados brutos em DataFrame"""
    
    # Mapear clubes
    clubes_dict = {int(k): v.get('nome', 'Desconhecido') for k, v in dados_clubes.items()}
    
    # Mapear partidas
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
    
    # Processar atletas
    atletas_lista = []
    atletas = dados_atletas.get('atletas', [])
    if isinstance(atletas, dict):
        atletas = list(atletas.values())
    
    for atleta in atletas:
        if atleta.get('status_id') != 7:  # Apenas prováveis
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


def filter_elite_players(df: pd.DataFrame, strategy: str):
    """Filtra apenas jogadores DE ELITE com histórico"""
    
    if strategy == 'POINTS':
        elite = (
            ((df['media'] >= 5.0) & (df['jogos'] >= 5)) |
            (df['preco'] >= 10.0)
        )
    else:  # WEALTH
        elite = (
            ((df['media'] >= 3.0) & (df['jogos'] >= 3)) |
            ((df['preco'] >= 6.0) & (df['jogos'] > 0))
        )
    
    df_elite = df[elite].copy()
    
    print(f"✅ Filtro ELITE: {len(df_elite)} de {len(df)} jogadores (Média: {df_elite['media'].mean():.2f})")
    
    return df_elite


def check_opponent_conflict(selected_players: List, new_player: pd.Series) -> bool:
    """Verifica conflitos de adversários diretos"""
    
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
            
            if (new_is_denfender and player_is_attacker) or (new_is_attacker and player_is_defender):
                return True
        
    
    return False


def pro_optimization(df: pd.DataFrame, budget: float, formation: str, score_col: str):
    """Otimização PRO sem conflitos de adversários"""
    
    formation_req = FORMATION_CONSTRAINTS[formation]
    
    df['cost_benefit'] = df[score_col] / (df['preco'] + 0.1)
    df_sorted = df.sort_values(score_col, ascending=False).copy()
    
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
    
    if len(selected) < 12:
        # Fallback: completar com custo-benefício
        df_cb = df.sort_values('cost_benefit', ascending=False).copy()
        
        for _, player in df_cb.iterrows():
            if len(selected) >= 12:
                break
            
            pos_id = player['posicao_id']
            price = player['preco']
            
            if pos_count[pos_id] >= formation_req[pos_id] or price > remaining_budget:
                continue
            
            if player['atleta_id'] not in [p['atleta_id'] for p in selected]:
                selected.append(player)
                remaining_budget -= price
                pos_count[pos_id] += 1
    
    return pd.DataFrame(selected)


def calculate_points_score_pro(df):
    """Score para estratégia POINTS"""
    media_weight = df['media'] * 3.0
    defense_bonus = df.get('defense_potential', 0) * 2.0
    experience_bonus = df['jogos'].clip(upper=10) * 0.5
    
    return media_weight + defense_bonus + experience_bonus


def calculate_wealth_score_pro(df):
    """Score para estratégia WEALTH"""
    base = df['media'].fillna(3.0)
    mpv = df['minimo_valorizar'].where(df['minimo_valorizar'] != 0, base * 0.7)
    
    valuation_ease = (base - mpv + 2).clip(lower=0)
    price_factor = 20.0 / (df['preco'] + 1.0)
    
    return (valuation_ease * 5) + (price_factor * 3)

# ========================================================================
# API CREDENTIALS
# ========================================================================

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY", "")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "")

# ========================================================================
# LAYER 1: RAPIDAPI - Forma dos Times (Dados em Tempo Real)
# ========================================================================

def buscar_forma_times_rapidapi():
    """
    Busca a forma recente dos times brasileiros via RapidAPI
    Retorna dict: {clube_nome: {'form': 'WWDWL', 'multiplier': 1.1}}
    """
    
    print("\n📡 Consultando RapidAPI - Forma dos Times...")
    
    if not RAPIDAPI_KEY:
        print("⚠️ RAPIDAPI_KEY não configurada. Usando multipliers padrão.")
        return {}
    
    url = "https://free-api-live-football-data.p.rapidapi.com/football-team-form"
    
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": "free-api-live-football-data.p.rapidapi.com"
    }
    
    # Mapeamento de clubes do Cartola para nomes da API
    clube_mapping = {
        'FLA': 'Flamengo', 'PAL': 'Palmeiras', 'SAO': 'Sao Paulo',
        'CAP': 'Athletico Paranaense', 'INT': 'Internacional',
        'GRE': 'Gremio', 'COR': 'Corinthians', 'FLU': 'Fluminense',
        'BAH': 'Bahia', 'BOT': 'Botafogo', 'CRU': 'Cruzeiro',
        'VAS': 'Vasco', 'CFC': 'Coritiba', 'FOR': 'Fortaleza',
        'CAM': 'Atletico Mineiro'
    }
    
    forma_times = {}
    
    try:
        # Fazer request para cada time
        for sigla, nome_completo in clube_mapping.items():
            querystring = {"team": nome_completo, "league": "Brasileirao"}
            
            response = requests.get(url, headers=headers, params=querystring, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                
                # Extrair forma (últimos 5 jogos)
                if 'form' in data:
                    form_str = data['form'][:5]  # WWDWL
                    
                    # Calcular multiplier baseado na forma
                    wins = form_str.count('W')
                    
                    if wins >= 4:
                        multiplier = 1.15  # 4+ vitórias = +15%
                    elif wins >= 3:
                        multiplier = 1.10  # 3 vitórias = +10%
                    elif wins >= 2:
                        multiplier = 1.05  # 2 vitórias = +5%
                    else:
                        multiplier = 1.00  # Forma ruim = neutro
                    
                    forma_times[sigla] = {
                        'form': form_str,
                        'multiplier': multiplier
                    }
                    
                    print(f"   ✅ {sigla}: {form_str} (x{multiplier:.2f})")
        
        if len(forma_times) > 0:
            print(f"✅ Forma de {len(forma_times)} times carregada!")
        else:
            print("⚠️ RapidAPI não retornou dados. Usando multipliers padrão.")
    
    except Exception as e:
        print(f"⚠️ Erro na RapidAPI: {e}")
        print("   Continuando com multipliers padrão (1.0x)")
    
    return forma_times


def aplicar_multipliers_forma(df: pd.DataFrame, forma_times: Dict) -> pd.DataFrame:
    """
    Aplica multipliers automáticos baseados na forma dos times
    """
    
    if not forma_times:
        return df
    
    print("\n⚡ Aplicando Multipliers de Forma...")
    
    def calc_multiplier(row):
        clube = row['clube']
        
        if clube in forma_times:
            return forma_times[clube]['multiplier']
        else:
            return 1.0
    
    df['form_multiplier'] = df.apply(calc_multiplier, axis=1)
    
    # Aplicar multiplier nas scores
    if 'pro_score' in df.columns:
        df['pro_score'] = df['pro_score'] * df['form_multiplier']
    
    times_boost = df[df['form_multiplier'] > 1.0]['clube'].unique()
    
    if len(times_boost) > 0:
        print(f"   ✅ {len(times_boost)} times com BOOST: {list(times_boost)}")
    
    return df


# ========================================================================
# LAYER 2: PERPLEXITY IA - Validação Inteligente da Escalação
# ========================================================================

def validar_escalacao_com_ia(lineup: pd.DataFrame, api_key: str) -> str:
    """
    Envia escalação para Perplexity IA validar:
    - Notícias de última hora (lesões, poupados)
    - Análise de matchups
    - Nota de segurança (0-10)
    - Sugestões de substituições
    """
    
    print("\n🤖 IA Analisando Escalação (Perplexity)...")
    
    # Serializar escalação em JSON rico
    escalacao_json = []
    
    for _, player in lineup.iterrows():
        escalacao_json.append({
            "posicao": player['posicao'],
            "jogador": player['apelido'],
            "clube": player['clube'],
            "adversario": player['opponent_name'],
            "casa": "Casa" if player['is_home_game'] else "Fora",
            "preco": float(player['preco']),
            "media": float(player['media'])
        })
    
    # System Prompt para a IA
    system_prompt = """Você é um Assistente Técnico especializado em Cartola FC.
    
Analise a escalação proposta e forneça:
1. Verificação de Notícias: Algum jogador está lesionado ou poupado?
2. Análise de Matchups: Confrontos favoráveis/desfavoráveis
3. Nota de Segurança: 0-10 (10 = escalação muito segura)
4. Sugestões: Se houver risco grave, sugira 1 substituto

Formato de resposta:
---
NOTÍCIAS: [resumo de lesões/poupados]
MATCHUPS: [análise dos confrontos]
NOTA: [0-10]/10
SUGESTÕES: [substituições se necessário]
---

Seja CONCISO (máximo 200 palavras)."""
    
    # Chamada para Perplexity API
    if not api_key:
        avg_media = lineup['media'].mean()
        jogadores_sem_hist = len(lineup[lineup['media'] == 0])
        return (
            "ANÁLISE AUTOMÁTICA (Sem IA):\n"
            "—\n"
            f"MÉDIA DO TIME: {avg_media:.2f}\n"
            f"NOTA: {min(10, max(0, int(avg_media * 1.5)))} / 10\n"
            f"SUGESTÕES: {jogadores_sem_hist} jogador(es) sem histórico\n"
            "—\n"
            "⚠️ Configure PERPLEXITY_API_KEY para ativar a validação de IA."
        )
    url = "https://api.perplexity.ai/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "llama-3.1-sonar-small-128k-online",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Analise esta escalação do Cartola FC:\n\n{json.dumps(escalacao_json, indent=2, ensure_ascii=False)}"}
        ],
        "max_tokens": 500,
        "temperature": 0.3,
        "top_p": 0.9
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            analise = data['choices'][0]['message']['content']
            
            print("✅ IA respondeu!")
            return analise
        else:
            print(f"⚠️ IA retornou erro {response.status_code}")
            return f"⚠️ Erro na API: {response.status_code}\n(Escalação gerada sem validação da IA)"
    
    except Exception as e:
        print(f"⚠️ Erro ao chamar IA: {e}")
        
        # Fallback: Análise baseada em estatísticas
        avg_media = lineup['media'].mean()
        jogadores_sem_hist = len(lineup[lineup['media'] == 0])
        
        analise_fallback = f"""
ANÁLISE AUTOMÁTICA (Sem IA):
---
NOTÍCIAS: Verificação manual recomendada
MATCHUPS: Média do time: {avg_media:.2f}
NOTA: {min(10, max(0, int(avg_media * 1.5)))}/10
SUGESTÕES: {jogadores_sem_hist} jogador(es) sem histórico - Considere substituir
---
⚠️ IA indisponível. Análise baseada apenas em estatísticas.
"""
        return analise_fallback


# ========================================================================
# PIPELINE ULTIMATE: ETL + RapidAPI + Otimização + IA
# ========================================================================

def pipeline_etl_com_rapidapi():
    """
    Pipeline completo que integra dados do Cartola + RapidAPI
    """
    
    print("\n" + "="*80)
    print("📡 PIPELINE DE DADOS - Cartola + RapidAPI")
    print("="*80)
    
    # 1. Dados do Cartola
    dados_atletas, dados_clubes, dados_partidas = load_cartola_data()
    
    if not dados_atletas:
        print("❌ Erro ao carregar dados do Cartola!")
        return None, None
    
    df = parse_data(dados_atletas, dados_clubes, dados_partidas)
    
    # 2. Forma dos times (RapidAPI)
    forma_times = buscar_forma_times_rapidapi()
    
    return df, forma_times


def rodar_solver_ultimate(df: pd.DataFrame, forma_times: Dict, 
                          budget: float, formation: str, strategy: str):
    """
    Solver ULTIMATE com forma dos times integrada
    """
    
    print("\n" + "="*80)
    print("🧮 SOLVER ULTIMATE - Otimização Matemática + IA")
    print("="*80)
    
    # Filtrar jogadores elite
    df = filter_elite_players(df, strategy)
    
    # Features básicas
    opp_strength = df.groupby('opponent_id')['media'].mean().to_dict()
    df['opponent_strength'] = df['opponent_id'].map(opp_strength).fillna(5.0)
    df['opponent_strength'] = 10 - df['opponent_strength']
    df['opponent_strength'] = df['opponent_strength'].clip(lower=0)
    
    # Defense potential
    def calc_defense_potential(row):
        if row['posicao_id'] in [1, 2, 3]:
            if row['is_home_game'] and row['opponent_strength'] > 5:
                return 5.0
        return 0.0
    
    df['defense_potential'] = df.apply(calc_defense_potential, axis=1)
    
    # Score baseado na estratégia
    if strategy == 'POINTS':
        df['pro_score'] = calculate_points_score_pro(df)
    else:
        df['pro_score'] = calculate_wealth_score_pro(df)
    
    # APLICAR MULTIPLIERS DE FORMA (RapidAPI)
    df = aplicar_multipliers_forma(df, forma_times)
    
    # Otimização PRO (sem conflitos)
    lineup = pro_optimization(df, budget, formation, 'pro_score')
    
    return lineup


# ========================================================================
# DISPLAY ULTIMATE
# ========================================================================

def display_ultimate_lineup(lineup, relatorio_ia, strategy, budget):
    """
    Display com análise da IA integrada
    """
    
    pos_order = {1: 1, 3: 2, 2: 3, 4: 4, 5: 5, 6: 6}
    lineup['pos_order'] = lineup['posicao_id'].map(pos_order)
    lineup = lineup.sort_values('pos_order')
    
    total_cost = lineup['preco'].sum()
    remaining = budget - total_cost
    avg_media = lineup['media'].mean()
    
    capitao_idx = lineup['media'].idxmax() if lineup['media'].max() > 0 else lineup.iloc[0].name
    capitao = lineup.loc[capitao_idx]
    
    print("\n" + "="*80)
    print("🏆 ESCALAÇÃO ULTIMATE (AI-POWERED)")
    print("="*80 + "\n")
    
    # Tabela da escalação
    display_df = lineup[['posicao', 'apelido', 'clube', 'opponent_name', 'is_home_game', 'preco', 'media']].copy()
    display_df.columns = ['Pos', 'Jogador', 'Clube', 'Adversário', 'Casa?', 'Preço', 'Média']
    display_df['Casa?'] = display_df['Casa?'].apply(lambda x: '🏠' if x else '✈️')
    display_df['Preço'] = display_df['Preço'].apply(lambda x: f"C${x:.1f}")
    
    # Destacar capitão
    display_df['Jogador'] = display_df.apply(
        lambda row: f"⭐{row['Jogador']}" if row['Jogador'] == capitao['apelido'] else row['Jogador'],
        axis=1
    )
    
    # Mostrar forma do time (se disponível)
    if 'form_multiplier' in lineup.columns:
        display_df['Boost'] = lineup['form_multiplier'].apply(lambda x: f"{x:.2f}x" if x > 1.0 else "")
    
    print(display_df.to_string(index=False))
    
    print("\n" + "-"*80)
    print(f"💰 CUSTO TOTAL: C$ {total_cost:.2f} / {budget:.2f}")
    print(f"💵 SOBROU: C$ {remaining:.2f}")
    print(f"📊 MÉDIA DO TIME: {avg_media:.2f}")
    print(f"⭐ CAPITÃO: {capitao['apelido']} (Média: {capitao['media']:.2f})")
    print("-"*80)
    
    # ANÁLISE DA IA (O pulo do gato!)
    print("\n" + "="*80)
    print("🤖 RELATÓRIO DE INTELIGÊNCIA ARTIFICIAL")
    print("="*80)
    print(relatorio_ia)
    print("="*80)
    
    return lineup


# ========================================================================
# MAIN ULTIMATE
# ========================================================================

def main():
    """
    Pipeline ULTIMATE completo
    """
    
    print("\n" + "="*80)
    print("🤖 CARTOLA FC - ULTIMATE OPTIMIZER (AI-POWERED)")
    print("="*80)
    print("✅ RapidAPI: Forma dos times em tempo real")
    print("✅ Perplexity IA: Validação automática de escalação")
    print("✅ Multipliers: Automáticos baseados em forma (WWWWW = 1.15x)")
    print("✅ Análise de Riscos: Lesões, poupados, matchups")
    print("="*80 + "\n")
    
    # Input do usuário
    budget = float(input("💰 Orçamento: "))
    
    print("\n📊 Estratégia:")
    print("   1 - PONTOS (Jogadores consolidados)")
    print("   2 - VALORIZAÇÃO (Jovens com potencial)")
    strategy_choice = input("Escolha (1 ou 2): ")
    strategy = 'POINTS' if strategy_choice == '1' else 'WEALTH'
    
    print("\n⚽ Formação:")
    print("   1 - 4-3-3  |  2 - 3-4-3  |  3 - 3-5-2  |  4 - 4-4-2")
    formation_choice = input("Escolha (1-4): ")
    formations = ['4-3-3', '3-4-3', '3-5-2', '4-4-2']
    formation = formations[int(formation_choice) - 1]
    
    print("\n" + "="*80)
    print("✅ CONFIGURAÇÃO")
    print(f"   💰 Orçamento: C$ {budget:.2f}")
    print(f"   📊 Estratégia: {strategy}")
    print(f"   ⚽ Formação: {formation}")
    print("="*80 + "\n")
    
    confirm = input("Iniciar? (S/N): ")
    if confirm.upper() != 'S':
        print("❌ Cancelado")
        return
    
    # ============================================================
    # PIPELINE AUTOMATIZADO COMPLETO
    # ============================================================
    
    # PASSO 1: ETL com RapidAPI
    df, forma_times = pipeline_etl_com_rapidapi()
    
    if df is None:
        print("❌ Erro fatal no pipeline!")
        return
    
    # PASSO 2: Otimização Matemática (com multipliers de forma)
    lineup = rodar_solver_ultimate(df, forma_times, budget, formation, strategy)
    
    if len(lineup) < 12:
        print("\n❌ Não foi possível gerar escalação completa!")
        return
    
    # PASSO 3: Validação com IA (Perplexity)
    relatorio_ia = validar_escalacao_com_ia(lineup, PERPLEXITY_API_KEY)
    
    # PASSO 4: Display Unificado (Tabela + Análise IA)
    final_lineup = display_ultimate_lineup(lineup, relatorio_ia, strategy, budget)
    
    # PASSO 5: Salvar
    output_file = f'ultimate_lineup_{strategy}_{formation.replace("-", "")}.xlsx'
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        final_lineup.to_excel(writer, sheet_name='Escalação Ultimate', index=False)
        
        # Adicionar análise da IA como segunda aba
        df_ia = pd.DataFrame({
            'Análise da IA': [relatorio_ia]
        })
        df_ia.to_excel(writer, sheet_name='Relatório IA', index=False)
    
    print(f"\n✅ Escalação salva em: {output_file}")
    print("\n" + "="*80)
    print("🏆 OTIMIZAÇÃO ULTIMATE CONCLUÍDA!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
