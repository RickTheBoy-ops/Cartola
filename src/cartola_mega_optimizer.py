#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
========================================================================
CARTOLA FC - MEGA OPTIMIZER (Todas as Variáveis do Brasileirão)
========================================================================
Sistema TOTALMENTE AUTOMÁTICO que usa TODAS as variáveis disponíveis
para gerar a melhor escalação possível via Programação Linear (PuLP).

Variáveis consideradas:
  📊 Média geral, última pontuação, variação, jogos disputados
  ⚽ Scouts detalhados (G, A, SG, DS, etc.)
  🏠 Mando de campo (Casa/Fora)
  💪 Força do adversário (ataque e defesa)
  🚫 Conflito de adversários (DEF vs ATK do mesmo jogo)
  💰 Mínimo para valorizar, preço, custo-benefício
  🎯 PuLP: Otimização Linear Inteira (solução matematicamente ótima)
  🔄 Testa TODAS as formações e escolhe a melhor

Execução: py src/cartola_mega_optimizer.py
========================================================================
"""

import pandas as pd
import numpy as np
import requests
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings('ignore')

try:
    from pulp import (LpProblem, LpMaximize, LpVariable, lpSum, LpStatus,
                      PULP_CBC_CMD, value)
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False
    print("⚠️ PuLP não instalado. Instale com: pip install pulp")

# ========================================================================
# CONFIGURAÇÕES
# ========================================================================

FORMATION_CONSTRAINTS = {
    '4-3-3': {1: 1, 2: 2, 3: 2, 4: 3, 5: 3, 6: 1},
    '3-4-3': {1: 1, 2: 0, 3: 3, 4: 4, 5: 3, 6: 1},
    '3-5-2': {1: 1, 2: 0, 3: 3, 4: 5, 5: 2, 6: 1},
    '4-4-2': {1: 1, 2: 2, 3: 2, 4: 4, 5: 2, 6: 1},
}

POS_MAP = {1: "GOL", 2: "LAT", 3: "ZAG", 4: "MEI", 5: "ATA", 6: "TEC"}
POS_NAME = {1: "Goleiro", 2: "Lateral", 3: "Zagueiro", 4: "Meia", 5: "Atacante", 6: "Técnico"}

# ========================================================================
# SISTEMA OFICIAL DE PONTUAÇÃO DO CARTOLA FC
# ========================================================================
# Valores EXATOS que o Cartola FC atribui a cada ação (scout)

ALL_SCOUTS = ['G','A','FT','FD','FF','FS','SG','DE','DP','DS','V','PS',
              'GC','CV','CA','GS','PP','PC','FC','I']

# Pontos oficiais do Cartola FC por ação
CARTOLA_OFFICIAL_POINTS = {
    'G':   8.0,   # Gol
    'A':   5.0,   # Assistência
    'FT':  3.5,   # Finalização na trave
    'FD':  1.2,   # Finalização defendida
    'FF':  0.8,   # Finalização pra fora
    'FS':  0.5,   # Falta sofrida
    'SG':  5.0,   # Jogo sem sofrer gol (GOL/ZAG/LAT)
    'DE':  1.0,   # Desarme
    'DP':  7.0,   # Defesa de pênalti
    'DS':  3.0,   # Defesa difícil (goleiro)
    'V':   1.0,   # Vitória do time  (estimativa)
    'PS':  0.8,   # Passe decisivo
    'GC': -5.0,   # Gol contra
    'CV': -5.0,   # Cartão vermelho
    'CA': -2.0,   # Cartão amarelo
    'GS': -2.0,   # Gol sofrido (GOL/ZAG/LAT)
    'PP': -4.0,   # Pênalti perdido
    'PC': -0.3,   # Passe errado (penalização leve)
    'FC': -0.5,   # Falta cometida
    'I':  -0.5,   # Impedimento
}

# ========================================================================
# PESOS POR POSIÇÃO — O que mais importa para cada posição
# Multiplicadores sobre o scout/jogo. Ex: para GOL, DS vale 2.0x extra
# ========================================================================

POSITION_SCOUT_MULTIPLIERS = {
    # GOLEIRO: defesas difíceis, SG e evitar gols sofridos são tudo
    1: {
        'DS': 2.5,   # Defesa difícil é o scout MAIS importante
        'SG': 2.0,   # Jogo sem sofrer gol = 5pts
        'DP': 2.0,   # Defesa de pênalti é raro mas vale muito
        'GS': 1.5,   # Gol sofrido pesa demais para goleiro
        'G':  0.3,   # Goleiro quase não faz gol
        'A':  0.3,
        'FT': 0.1, 'FD': 0.1, 'FF': 0.1,
        'DE': 0.5,
        'FS': 0.3,
        'FC': 1.0, 'CA': 1.5, 'CV': 2.0,
        'PC': 0.5, 'I': 0.1, 'GC': 2.0, 'PP': 1.0,
        'V': 1.5, 'PS': 0.3,
    },
    # LATERAL: equilíbrio entre defesa (SG, DE) e ataque (A, FS, cruzamentos)
    2: {
        'A':  2.0,   # Assistências são o forte do lateral ofensivo
        'SG': 1.8,   # SG importante mas lateral sofre mais
        'FS': 1.5,   # Falta sofrida no ataque
        'DE': 1.5,   # Desarme na marcação
        'DS': 0.5,   # Lateral não faz defesa difícil
        'G':  1.5,   # Gol de lateral é raro mas vale muito
        'FT': 1.2, 'FD': 1.0, 'FF': 0.8,
        'GS': 1.3,   # Gol sofrido afeta lateral
        'DP': 0.1,
        'FC': 1.2, 'CA': 1.5, 'CV': 2.0,
        'PC': 0.8, 'I': 0.3, 'GC': 1.5, 'PP': 1.0,
        'V': 1.2, 'PS': 1.5,
    },
    # ZAGUEIRO: desarme, SG e evitar gols sofridos
    3: {
        'SG': 2.0,   # Jogo sem sofrer gol é crucial
        'DE': 2.0,   # Desarme é o forte do zagueiro
        'G':  1.5,   # Gol de zagueiro (cabeçada) é valioso
        'GS': 1.8,   # Gol sofrido penaliza muito
        'DS': 0.3,   # Zagueiro não faz defesa difícil
        'A':  1.0,
        'FT': 1.0, 'FD': 0.8, 'FF': 0.5,
        'FS': 0.8,
        'DP': 0.1,
        'FC': 1.5,   # Zagueiro faz muita falta → risco
        'CA': 2.0,   # Cartão amarelo é risco ALTO para zagueiro
        'CV': 2.5,   # Cartão vermelho = desastre
        'PC': 0.8, 'I': 0.1, 'GC': 2.0, 'PP': 1.0,
        'V': 1.3, 'PS': 0.5,
    },
    # MEIA: gols, assistências, finalizações e passes decisivos
    4: {
        'G':  2.0,   # Gol de meia é muito comum no Cartola
        'A':  2.0,   # Assistência é o forte do meia
        'FT': 1.8,   # Finalização na trave = quase gol
        'FD': 1.5,   # Finalização defendida = pressão ofensiva
        'FF': 1.0,   # Finalização pra fora (tenta muito)
        'FS': 1.2,   # Falta sofrida no meio
        'PS': 2.0,   # Passe decisivo é crucial para armadores
        'DE': 1.2,   # Meia que desarma pontua extra
        'SG': 0.3,   # SG não afeta meia
        'DS': 0.1,
        'GS': 0.2,
        'DP': 0.1,
        'FC': 1.0, 'CA': 1.2, 'CV': 1.5,
        'PC': 1.0, 'I': 0.5, 'GC': 1.0, 'PP': 2.0,
        'V': 1.0,
    },
    # ATACANTE: gols, finalizações e assistências
    5: {
        'G':  2.5,   # GOL é TUDO para atacante
        'A':  1.5,   # Assistência também
        'FT': 2.0,   # Finalização na trave → pressão
        'FD': 1.8,   # Finalização defendida → tenta bastante
        'FF': 1.2,   # Finalização pra fora → volume
        'FS': 1.0,   # Falta sofrida na área
        'PS': 1.0,
        'DE': 0.5,   # Atacante não desarma
        'SG': 0.1,
        'DS': 0.1,
        'GS': 0.1,
        'DP': 0.1,
        'FC': 0.8, 'CA': 1.0, 'CV': 1.5,
        'PC': 0.5, 'I': 1.5,  # Impedimento é risco de atacante
        'GC': 0.5, 'PP': 2.5, # Pênalti perdido = desastre para ATA
        'V': 1.0,
    },
    # TÉCNICO: resultado do time importa mais
    6: {
        'V': 3.0,   # Vitória do time
        'SG': 1.5,  # Time não sofrer gol
        'G': 0.5, 'A': 0.5, 'FT': 0.3, 'FD': 0.2, 'FF': 0.1,
        'FS': 0.2, 'DE': 0.3, 'DS': 0.3, 'DP': 0.3,
        'GS': 1.0, 'FC': 0.2, 'CA': 0.3, 'CV': 0.5,
        'PC': 0.2, 'I': 0.1, 'GC': 0.5, 'PP': 0.3, 'PS': 0.3,
    }
}


# ========================================================================
# ETAPA 1: CARREGAR DADOS (APIs do Cartola)
# ========================================================================

def load_all_data():
    """Carrega TODOS os dados disponíveis das APIs do Cartola FC"""

    API_ATLETAS = "https://api.cartola.globo.com/atletas/mercado"
    API_CLUBES = "https://api.cartola.globo.com/clubes"
    API_PARTIDAS = "https://api.cartola.globo.com/partidas"
    API_RODADA = "https://api.cartola.globo.com/mercado/status"

    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    print("📡 ETAPA 1: Carregando dados do Cartola FC...")

    try:
        r1 = requests.get(API_ATLETAS, headers=HEADERS, timeout=30)
        r2 = requests.get(API_CLUBES, headers=HEADERS, timeout=30)
        r3 = requests.get(API_PARTIDAS, headers=HEADERS, timeout=30)

        dados_atletas = r1.json()
        dados_clubes = r2.json()
        dados_partidas = r3.json()

        # Tentar pegar status do mercado
        try:
            r4 = requests.get(API_RODADA, headers=HEADERS, timeout=10)
            dados_mercado = r4.json()
            rodada = dados_mercado.get('rodada_atual', '?')
            status = dados_mercado.get('status_mercado', '?')
            status_map = {1: "Aberto", 2: "Fechado", 3: "Em manutenção", 4: "Apuração"}
            print(f"   📅 Rodada: {rodada} | Mercado: {status_map.get(status, status)}")
        except:
            rodada = '?'

        num_atletas = len(dados_atletas.get('atletas', []))
        print(f"   ✅ {num_atletas} atletas carregados")
        print(f"   ✅ {len(dados_clubes)} clubes carregados")

        return dados_atletas, dados_clubes, dados_partidas

    except Exception as e:
        print(f"   ❌ Erro ao carregar: {e}")
        return None, None, None


# ========================================================================
# ETAPA 2: PROCESSAR E ENRIQUECER DADOS
# ========================================================================

def process_all_data(dados_atletas, dados_clubes, dados_partidas):
    """
    Processa dados brutos em DataFrame RICO com todas as variáveis.
    """

    print("\n⚙️  ETAPA 2: Processando e enriquecendo dados...")

    # 1. Mapear clubes
    clubes_dict = {}
    clubes_abrev = {}
    for k, v in dados_clubes.items():
        cid = int(k)
        clubes_dict[cid] = v.get('nome', 'Desc.')
        clubes_abrev[cid] = v.get('abreviacao', v.get('nome', '?')[:3].upper())

    # 2. Mapear partidas
    partidas_dict = {}
    partidas_lista = []

    if isinstance(dados_partidas, dict):
        for key, val in dados_partidas.items():
            if isinstance(val, dict) and 'partidas' in val:
                partidas_lista.extend(val['partidas'])
            elif isinstance(val, list):
                partidas_lista.extend(val)
            elif isinstance(val, dict) and 'clube_casa_id' in val:
                partidas_lista.append(val)

    for partida in partidas_lista:
        if not isinstance(partida, dict):
            continue
        casa_id = partida.get('clube_casa_id')
        vis_id = partida.get('clube_visitante_id')
        if casa_id and vis_id:
            partidas_dict[casa_id] = {
                'opponent_id': vis_id,
                'opponent_name': clubes_abrev.get(vis_id, '?'),
                'is_home': True
            }
            partidas_dict[vis_id] = {
                'opponent_id': casa_id,
                'opponent_name': clubes_abrev.get(casa_id, '?'),
                'is_home': False
            }

    # 3. Processar atletas com TODOS os campos
    atletas_raw = dados_atletas.get('atletas', [])
    if isinstance(atletas_raw, dict):
        atletas_raw = list(atletas_raw.values())

    rows = []
    for a in atletas_raw:
        if a.get('status_id') != 7:  # Apenas status "Provável"
            continue

        cid = a.get('clube_id')
        info_p = partidas_dict.get(cid, {})

        # Scouts do jogador
        scouts = a.get('scout', {}) or {}

        row = {
            'atleta_id': a.get('atleta_id'),
            'apelido': a.get('apelido', '?'),
            'posicao_id': a.get('posicao_id'),
            'posicao': POS_MAP.get(a.get('posicao_id'), '?'),
            'clube_id': cid,
            'clube': clubes_abrev.get(cid, '?'),
            'clube_nome': clubes_dict.get(cid, '?'),
            'preco': float(a.get('preco_num', 0)),
            'media': float(a.get('media_num', 0)),
            'variacao': float(a.get('variacao_num', 0)),
            'ultima_pontuacao': float(a.get('pontos_num', 0)),
            'jogos': int(a.get('jogos_num', 0)),
            'minimo_valorizar': float(a.get('minimo_para_valorizar', 0)),
            'opponent_id': info_p.get('opponent_id'),
            'opponent_name': info_p.get('opponent_name', 'N/A'),
            'is_home': info_p.get('is_home', False),
        }

        # Adicionar scouts individuais
        for scout_key in ALL_SCOUTS:
            row[f'scout_{scout_key}'] = float(scouts.get(scout_key, 0))

        rows.append(row)

    df = pd.DataFrame(rows)
    print(f"   ✅ {len(df)} atletas prováveis com scouts processados")

    return df, partidas_dict


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature Engineering V2: Análise COMPLETA por posição.
    Usa pesos específicos para cada posição, sistema oficial de pontos,
    análise de disciplina, eficiência de finalização e probabilidade de SG.
    """

    print("\n🧪 ETAPA 3: Feature Engineering V2 (ANÁLISE POR POSIÇÃO)...")

    df = df.copy()
    jogos_safe = df['jogos'].clip(lower=1)

    # ---------------------------------------------------------------
    # 1. SCOUT PER GAME: Normalizar todos os scouts por jogo disputado
    # ---------------------------------------------------------------
    for scout_key in ALL_SCOUTS:
        col = f'scout_{scout_key}'
        if col in df.columns:
            df[f'spg_{scout_key}'] = df[col] / jogos_safe  # scout per game

    # ---------------------------------------------------------------
    # 2. PONTUAÇÃO ESPERADA POR POSIÇÃO (Position-Weighted Scout Score)
    #    Simula quantos PONTOS DO CARTOLA o jogador geraria por jogo
    # ---------------------------------------------------------------
    def calc_position_score(row):
        pos_id = row['posicao_id']
        multipliers = POSITION_SCOUT_MULTIPLIERS.get(pos_id, {})
        score = 0.0
        for scout_key, official_pts in CARTOLA_OFFICIAL_POINTS.items():
            spg_col = f'spg_{scout_key}'
            if spg_col in row.index:
                spg = row[spg_col]
                pos_mult = multipliers.get(scout_key, 1.0)
                score += spg * official_pts * pos_mult
        return score

    df['position_score'] = df.apply(calc_position_score, axis=1)

    # ---------------------------------------------------------------
    # 3. EFICIÊNCIA DE FINALIZAÇÃO (para MEI e ATA)
    #    Gols / (FT + FD + FF + G) = taxa de conversão
    # ---------------------------------------------------------------
    total_finalizacoes = (
        df.get('scout_G', 0) + df.get('scout_FT', 0)
        + df.get('scout_FD', 0) + df.get('scout_FF', 0)
    )
    # Evitar divisão por zero
    df['total_fin'] = total_finalizacoes.clip(lower=0)
    df['fin_efficiency'] = 0.0
    mask_fin = df['total_fin'] > 0
    df.loc[mask_fin, 'fin_efficiency'] = (
        df.loc[mask_fin, 'scout_G'] / df.loc[mask_fin, 'total_fin']
    )
    # Bônus para quem finaliza MUITO (volume) E converte
    df['fin_volume_pg'] = df['total_fin'] / jogos_safe
    df['fin_bonus'] = 0.0
    # MEI e ATA ganham bônus por volume alto de finalizações + eficiência
    mask_atk = df['posicao_id'].isin([4, 5])
    df.loc[mask_atk, 'fin_bonus'] = (
        df.loc[mask_atk, 'fin_volume_pg'] * 2.0
        + df.loc[mask_atk, 'fin_efficiency'] * 5.0
    )

    print(f"   ⚽ Finalização: avg volume/jogo={df.loc[mask_atk, 'fin_volume_pg'].mean():.2f} | "
          f"eficiência={df.loc[mask_atk, 'fin_efficiency'].mean():.1%}")

    # ---------------------------------------------------------------
    # 4. DISCIPLINA: Risco de cartões e faltas
    #    Desconto para jogadores indisciplinados
    # ---------------------------------------------------------------
    df['ca_per_game'] = df.get('scout_CA', 0) / jogos_safe
    df['cv_per_game'] = df.get('scout_CV', 0) / jogos_safe
    df['fc_per_game'] = df.get('scout_FC', 0) / jogos_safe

    # Risco de disciplina (quanto MAIOR, PIOR)
    df['discipline_risk'] = (
        df['ca_per_game'] * 2.0    # -2 pts por amarelo
        + df['cv_per_game'] * 5.0  # -5 pts por vermelho
        + df['fc_per_game'] * 0.5  # -0.5 por falta
    )
    # Multiplicador: jogador limpo = 1.0, sujo = 0.85
    df['discipline_mult'] = (1.0 - df['discipline_risk'] * 0.1).clip(0.85, 1.05)
    # Bônus para quem NUNCA tomou cartão
    no_cards = (df.get('scout_CA', 0) == 0) & (df.get('scout_CV', 0) == 0)
    df.loc[no_cards, 'discipline_mult'] = df.loc[no_cards, 'discipline_mult'] + 0.03

    print(f"   🟨 Disciplina: {(df['ca_per_game'] > 0.3).sum()} jogadores com risco alto de cartão")

    # ---------------------------------------------------------------
    # 5. DESARME SCORE (para ZAG, LAT e MEI defensivos)
    # ---------------------------------------------------------------
    df['de_per_game'] = df.get('scout_DE', 0) / jogos_safe
    df['desarme_bonus'] = 0.0
    mask_def = df['posicao_id'].isin([2, 3])
    df.loc[mask_def, 'desarme_bonus'] = df.loc[mask_def, 'de_per_game'] * 3.0
    # Meia que desarma também ganha bônus (meia-volante)
    mask_mei = df['posicao_id'] == 4
    df.loc[mask_mei, 'desarme_bonus'] = df.loc[mask_mei, 'de_per_game'] * 1.5

    # ---------------------------------------------------------------
    # 6. PROBABILIDADE DE SALDO DE GOL (SG)
    #    GOL/ZAG/LAT ganham 5pts por SG - prever chance de SG
    # ---------------------------------------------------------------
    df['sg_per_game'] = df.get('scout_SG', 0) / jogos_safe
    df['gs_per_game'] = df.get('scout_GS', 0) / jogos_safe
    df['sg_probability'] = 0.0
    mask_sg_pos = df['posicao_id'].isin([1, 2, 3])
    # SG probabilidade baseada no histórico
    df.loc[mask_sg_pos, 'sg_probability'] = df.loc[mask_sg_pos, 'sg_per_game']
    # Valor esperado do SG = probabilidade * 5 pontos
    df['sg_expected_pts'] = df['sg_probability'] * 5.0
    # Penalidade por gols sofridos frequentes
    df['gs_penalty'] = 0.0
    df.loc[mask_sg_pos, 'gs_penalty'] = df.loc[mask_sg_pos, 'gs_per_game'] * 2.0

    print(f"   🛡️ Saldo de Gol: {(df['sg_per_game'] > 0.3).sum()} jogadores com SG freq. > 30%")

    # ---------------------------------------------------------------
    # 7. ASSISTÊNCIA SCORE (LAT e MEI armadores)
    # ---------------------------------------------------------------
    df['a_per_game'] = df.get('scout_A', 0) / jogos_safe
    df['assist_bonus'] = 0.0
    mask_assist = df['posicao_id'].isin([2, 4])  # Laterais e meias
    df.loc[mask_assist, 'assist_bonus'] = df.loc[mask_assist, 'a_per_game'] * 5.0

    # ---------------------------------------------------------------
    # 8. GOLEIRO SCORE ESPECIAL (DS por jogo é crucial)
    # ---------------------------------------------------------------
    df['ds_per_game'] = df.get('scout_DS', 0) / jogos_safe
    df['gk_special'] = 0.0
    mask_gol = df['posicao_id'] == 1
    df.loc[mask_gol, 'gk_special'] = (
        df.loc[mask_gol, 'ds_per_game'] * 6.0    # Defesas difíceis
        + df.loc[mask_gol, 'sg_expected_pts'] * 2.0  # SG
        - df.loc[mask_gol, 'gs_penalty'] * 1.0   # Gols sofridos
    )

    # ---------------------------------------------------------------
    # 9. MOMENTUM: Forma recente (última pontuação vs média)
    # ---------------------------------------------------------------
    df['momentum'] = 1.0
    mask = df['media'] > 0
    df.loc[mask, 'momentum'] = (
        df.loc[mask, 'ultima_pontuacao'] / df.loc[mask, 'media']
    ).clip(0.5, 2.0)

    # ---------------------------------------------------------------
    # 10. CONSISTENCY: Mais jogos = mais confiável
    # ---------------------------------------------------------------
    max_jogos = max(df['jogos'].max(), 1)
    df['consistency'] = (df['jogos'] / max_jogos).clip(0, 1)

    # ---------------------------------------------------------------
    # 11. HOME ADVANTAGE: +12% para mandantes
    # ---------------------------------------------------------------
    df['home_bonus'] = df['is_home'].apply(lambda x: 1.12 if x else 1.0)

    # ---------------------------------------------------------------
    # 12. OPPONENT STRENGTH por posição
    # ---------------------------------------------------------------
    # Gols por jogo do adversário (força ofensiva)
    atk_players = df[df['posicao_id'].isin([4, 5])]
    opp_attack_strength = atk_players.groupby('clube_id')['media'].mean().to_dict()
    # Média defensiva do adversário
    def_players = df[df['posicao_id'].isin([1, 2, 3])]
    opp_defense_strength = def_players.groupby('clube_id')['media'].mean().to_dict()
    # Gols sofridos pelo adversário (GS total por time)
    gs_by_team = df.groupby('clube_id')['scout_GS'].sum()
    gs_per_team_game = (gs_by_team / df.groupby('clube_id')['jogos'].max().clip(lower=1)).to_dict()

    def calc_opp_bonus(row):
        opp_id = row['opponent_id']
        pos_id = row['posicao_id']
        if pd.isna(opp_id) or opp_id is None:
            return 1.0
        opp_id = int(opp_id)

        if pos_id in [1, 2, 3]:  # Defensor: adversário com ataque fraco
            opp_off = opp_attack_strength.get(opp_id, 5.0)
            bonus = 1.0 + max(0, (6.0 - opp_off) * 0.04)
        elif pos_id in [4, 5]:  # Atacante: adversário com defesa fraca
            opp_def = opp_defense_strength.get(opp_id, 5.0)
            # Também considerar gols sofridos pelo adversário
            gs_opp = gs_per_team_game.get(opp_id, 1.0)
            bonus = 1.0 + max(0, (6.0 - opp_def) * 0.03) + gs_opp * 0.02
        else:
            avg = (opp_attack_strength.get(opp_id, 5.0) + opp_defense_strength.get(opp_id, 5.0)) / 2
            bonus = 1.0 + max(0, (5.0 - avg) * 0.02)
        return min(bonus, 1.25)

    df['opp_bonus'] = df.apply(calc_opp_bonus, axis=1)

    # ---------------------------------------------------------------
    # 13. TREND: Variação de preço indica forma
    # ---------------------------------------------------------------
    df['trend_bonus'] = 1.0
    df.loc[df['variacao'] > 0, 'trend_bonus'] = 1.05
    df.loc[df['variacao'] > 2, 'trend_bonus'] = 1.10
    df.loc[df['variacao'] < -2, 'trend_bonus'] = 0.92  # Desvalorizando = ruim

    # ---------------------------------------------------------------
    # 14. FALTA SOFRIDA SCORE (para MEI e ATA)
    # ---------------------------------------------------------------
    df['fs_per_game'] = df.get('scout_FS', 0) / jogos_safe
    df['fs_bonus'] = 0.0
    mask_fs = df['posicao_id'].isin([4, 5])
    df.loc[mask_fs, 'fs_bonus'] = df.loc[mask_fs, 'fs_per_game'] * 1.5

    # ===============================================================
    # MEGA SCORE V2: Composição FINAL com todas as variáveis
    # ===============================================================

    df['mega_score'] = (
        # Base: Média oficial do Cartola (peso forte)
        df['media'] * 3.0

        # Score posicional (scouts ponderados por posição)
        + df['position_score'] * 2.0

        # Bônus específicos por posição
        + df['fin_bonus'] * 1.0        # Finalização (MEI/ATA)
        + df['desarme_bonus'] * 1.0    # Desarme (ZAG/LAT)
        + df['sg_expected_pts'] * 1.5  # SG esperado (GOL/ZAG/LAT)
        + df['assist_bonus'] * 1.0     # Assistências (LAT/MEI)
        + df['gk_special'] * 1.0       # Especial goleiro
        + df['fs_bonus'] * 0.5         # Falta sofrida (MEI/ATA)

        # Penalidades
        - df['gs_penalty'] * 1.0       # Gols sofridos

        # Consistência
        + df['consistency'] * 2.0
    )

    # Multiplicadores contextuais
    df['mega_score'] = (
        df['mega_score']
        * df['home_bonus']          # Mando de campo
        * df['opp_bonus']           # Fraqueza do adversário
        * df['momentum'].clip(0.85, 1.25)  # Forma recente
        * df['trend_bonus']         # Tendência de preço
        * df['discipline_mult']     # Desconto por indisciplina
    )

    # Filtrar sem jogos
    df_valid = df[df['jogos'] >= 1].copy()

    print(f"\n   ✅ {len(df_valid)} atletas com dados válidos")
    print(f"   📊 Mega Score V2: min={df_valid['mega_score'].min():.1f} | "
          f"avg={df_valid['mega_score'].mean():.1f} | "
          f"max={df_valid['mega_score'].max():.1f}")

    # Top 3 por posição
    print(f"\n   🏆 TOP 3 POR POSIÇÃO (MegaScore):")
    for pos_id in [1, 2, 3, 4, 5, 6]:
        pos_df = df_valid[df_valid['posicao_id'] == pos_id].nlargest(3, 'mega_score')
        nomes = ', '.join([f"{r['apelido']}({r['mega_score']:.0f})" for _, r in pos_df.iterrows()])
        print(f"   {POS_MAP[pos_id]}: {nomes}")

    return df_valid


# ========================================================================
# ETAPA 4: OTIMIZAÇÃO COM PULP (PROGRAMAÇÃO LINEAR INTEIRA)
# ========================================================================

def get_conflict_pairs(df: pd.DataFrame) -> List[Tuple[int, int]]:
    """
    Identifica pares de jogadores com conflito de adversários.
    Regra: NÃO escalar defensor de um time + atacante/meia do adversário direto.
    (Ex: Zagueiro do PAL + Atacante do FLU quando PAL x FLU = conflito)
    Permite outros pares (ex: MEI vs MEI, ATA vs ATA de adversários).
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

            # Conflito apenas: defensor vs atacante/meia
            pi_def = pi['posicao_id'] in [1, 2, 3]
            pj_def = pj['posicao_id'] in [1, 2, 3]
            pi_atk = pi['posicao_id'] in [4, 5]
            pj_atk = pj['posicao_id'] in [4, 5]

            if (pi_def and pj_atk) or (pi_atk and pj_def):
                conflicts.append((idx_i, idx_j))

    return conflicts


def optimize_with_pulp(df: pd.DataFrame, budget: float, formation: str) -> Optional[pd.DataFrame]:
    """
    Otimização com PuLP: encontra a escalação MATEMATICAMENTE ÓTIMA.
    """

    if not PULP_AVAILABLE:
        print("   ❌ PuLP não disponível!")
        return None

    formation_req = FORMATION_CONSTRAINTS[formation]

    # Criar variáveis de decisão (0 ou 1 para cada jogador)
    prob = LpProblem(f"CartolaMega_{formation}", LpMaximize)

    player_vars = {}
    for idx in df.index:
        player_vars[idx] = LpVariable(f"x_{idx}", cat='Binary')

    # OBJETIVO: Maximizar mega_score total
    prob += lpSum(df.loc[idx, 'mega_score'] * player_vars[idx] for idx in df.index)

    # RESTRIÇÃO 1: Respeitar formação (quantidade por posição)
    for pos_id, count in formation_req.items():
        pos_indices = df[df['posicao_id'] == pos_id].index
        prob += lpSum(player_vars[idx] for idx in pos_indices) == count

    # RESTRIÇÃO 2: Total de 12 jogadores
    prob += lpSum(player_vars[idx] for idx in df.index) == 12

    # RESTRIÇÃO 3: Orçamento
    prob += lpSum(df.loc[idx, 'preco'] * player_vars[idx] for idx in df.index) <= budget

    # RESTRIÇÃO 4: Máximo de 3 jogadores por clube
    for clube_id in df['clube_id'].unique():
        clube_indices = df[df['clube_id'] == clube_id].index
        prob += lpSum(player_vars[idx] for idx in clube_indices) <= 3

    # RESTRIÇÃO 5: Conflitos de adversários (DEF vs ATK do mesmo jogo)
    conflicts = get_conflict_pairs(df)
    for idx_i, idx_j in conflicts:
        prob += player_vars[idx_i] + player_vars[idx_j] <= 1

    # Resolver
    solver = PULP_CBC_CMD(msg=0, timeLimit=30)
    prob.solve(solver)

    if LpStatus[prob.status] != 'Optimal':
        return None

    # Extrair solução
    selected_indices = [idx for idx in df.index if value(player_vars[idx]) == 1]
    lineup = df.loc[selected_indices].copy()

    return lineup


def find_best_lineup(df: pd.DataFrame, budget: float) -> Tuple[pd.DataFrame, str]:
    """
    Testa TODAS as formações e retorna a melhor escalação.
    """

    print(f"\n🧮 ETAPA 4: Otimização (PuLP) | Orçamento: C$ {budget:.2f}")
    print(f"   Testando todas as formações...")

    best_lineup = None
    best_formation = None
    best_score = -1

    for formation, reqs in FORMATION_CONSTRAINTS.items():
        lineup = optimize_with_pulp(df, budget, formation)

        if lineup is not None and len(lineup) == 12:
            total_score = lineup['mega_score'].sum()
            total_cost = lineup['preco'].sum()
            print(f"   ⚽ {formation}: Score={total_score:.1f} | Custo=C${total_cost:.1f}")

            if total_score > best_score:
                best_score = total_score
                best_lineup = lineup
                best_formation = formation
        else:
            print(f"   ❌ {formation}: Inviável com este orçamento")

    if best_lineup is not None:
        print(f"\n   🏆 MELHOR FORMAÇÃO: {best_formation} (Score: {best_score:.1f})")
    else:
        print("\n   ❌ Nenhuma formação viável encontrada!")

    return best_lineup, best_formation


# ========================================================================
# ETAPA 5: DISPLAY E EXPORTAÇÃO
# ========================================================================

def select_captain(lineup: pd.DataFrame) -> str:
    """Seleciona o melhor capitão (maior média com bom momento)"""

    lineup = lineup.copy()

    # Score de capitão: prioriza média alta + momentum bom + jogando em casa
    lineup['cap_score'] = (
        lineup['media'] * 2.0
        + lineup['ultima_pontuacao'] * 0.5
        + lineup['is_home'].astype(float) * 1.0
        + lineup['momentum'] * 0.5
    )

    cap_idx = lineup['cap_score'].idxmax()
    return lineup.loc[cap_idx, 'apelido']


def display_and_save(lineup: pd.DataFrame, formation: str, budget: float, df_all: pd.DataFrame):
    """Display completo e salva Excel"""

    # Ordenar por posição
    pos_order = {1: 1, 3: 2, 2: 3, 4: 4, 5: 5, 6: 6}
    lineup = lineup.copy()
    lineup['pos_order'] = lineup['posicao_id'].map(pos_order)
    lineup = lineup.sort_values('pos_order')

    total_cost = lineup['preco'].sum()
    remaining = budget - total_cost
    avg_media = lineup['media'].mean()
    total_mega = lineup['mega_score'].sum()

    capitao = select_captain(lineup)

    # ============================================================
    # DISPLAY
    # ============================================================

    print("\n" + "=" * 90)
    print("🏆 MEGA ESCALAÇÃO - MELHOR LINEUP POSSÍVEL (TODAS AS VARIÁVEIS)")
    print("=" * 90)
    print(f"⚽ Formação: {formation} | 📅 {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    print("=" * 90 + "\n")

    # Header
    header = f"{'Pos':<5} {'Jogador':<20} {'Clube':<6} {'vs':<6} {'Casa':<5} " \
             f"{'Preço':>7} {'Média':>6} {'Ùlt.Pts':>7} {'MegaScore':>10}"
    print(header)
    print("-" * 90)

    for _, p in lineup.iterrows():
        is_cap = "⭐" if p['apelido'] == capitao else "  "
        casa = "🏠" if p['is_home'] else "✈️"
        nome = f"{is_cap}{p['apelido']}"[:20]
        print(f"{p['posicao']:<5} {nome:<20} {p['clube']:<6} {p['opponent_name']:<6} "
              f"{casa:<5} C${p['preco']:>5.1f} {p['media']:>6.2f} "
              f"{p['ultima_pontuacao']:>7.2f} {p['mega_score']:>10.1f}")

    print("-" * 90)
    print(f"💰 CUSTO TOTAL: C$ {total_cost:.2f} / {budget:.2f} "
          f"(sobra C$ {remaining:.2f})")
    print(f"📊 MÉDIA DO TIME: {avg_media:.2f}")
    print(f"🎯 MEGA SCORE TOTAL: {total_mega:.1f}")
    print(f"⭐ CAPITÃO: {capitao}")
    print("-" * 90)

    # Estatísticas extras
    home_count = lineup['is_home'].sum()
    print(f"\n📈 ANÁLISE DA ESCALAÇÃO:")
    print(f"   🏠 Jogadores em casa: {home_count}/{len(lineup)}")
    print(f"   🔥 Momentum médio: {lineup['momentum'].mean():.2f}x")
    print(f"   💪 Bônus adversário médio: {lineup['opp_bonus'].mean():.2f}x")
    print(f"   📊 Consistência média: {lineup['consistency'].mean():.1%}")

    # Clubes usados
    clubes_used = lineup.groupby('clube')['apelido'].apply(list).to_dict()
    print(f"\n   ⚽ Clubes ({len(clubes_used)}): ", end="")
    for c, players in clubes_used.items():
        print(f"{c}({len(players)}) ", end="")
    print()

    # Alternativas (top 5 que ficaram de fora)
    selected_ids = set(lineup['atleta_id'])
    alternativas = df_all[~df_all['atleta_id'].isin(selected_ids)].nlargest(5, 'mega_score')

    if len(alternativas) > 0:
        print(f"\n🔄 TOP 5 ALTERNATIVAS (ficaram de fora):")
        print(f"   {'Jogador':<20} {'Pos':<5} {'Clube':<6} {'Média':>6} {'MegaScore':>10}")
        for _, a in alternativas.iterrows():
            print(f"   {a['apelido']:<20} {a['posicao']:<5} {a['clube']:<6} "
                  f"{a['media']:>6.2f} {a['mega_score']:>10.1f}")

    print("\n" + "=" * 90)

    # ============================================================
    # SALVAR EXCEL
    # ============================================================

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"mega_escalacao_{timestamp}.xlsx"

    # Preparar abas
    export_lineup = lineup[[
        'posicao', 'apelido', 'clube', 'opponent_name', 'is_home',
        'preco', 'media', 'ultima_pontuacao', 'variacao', 'jogos',
        'mega_score', 'momentum', 'opp_bonus', 'consistency'
    ]].copy()
    export_lineup.columns = [
        'Posição', 'Jogador', 'Clube', 'Adversário', 'Mandante',
        'Preço', 'Média', 'Última Pts', 'Variação', 'Jogos',
        'MegaScore', 'Momentum', 'Bônus Adv.', 'Consistência'
    ]
    export_lineup['Capitão'] = export_lineup['Jogador'].apply(
        lambda x: '⭐ SIM' if x == capitao else ''
    )

    # Resumo
    resumo = pd.DataFrame({
        'Info': [
            'Formação', 'Custo Total', 'Orçamento', 'Sobra',
            'Média do Time', 'MegaScore Total', 'Capitão',
            'Jogadores em Casa', 'Data/Hora'
        ],
        'Valor': [
            formation, f'C$ {total_cost:.2f}', f'C$ {budget:.2f}',
            f'C$ {remaining:.2f}', f'{avg_media:.2f}', f'{total_mega:.1f}',
            capitao, f'{home_count}/{len(lineup)}',
            datetime.now().strftime('%d/%m/%Y %H:%M')
        ]
    })

    # Top alternativas
    export_alt = alternativas[[
        'posicao', 'apelido', 'clube', 'opponent_name', 'preco',
        'media', 'mega_score'
    ]].copy()
    export_alt.columns = [
        'Posição', 'Jogador', 'Clube', 'Adversário',
        'Preço', 'Média', 'MegaScore'
    ]

    with pd.ExcelWriter(str(output_file), engine='openpyxl') as writer:
        export_lineup.to_excel(writer, sheet_name='Escalação', index=False)
        resumo.to_excel(writer, sheet_name='Resumo', index=False)
        export_alt.to_excel(writer, sheet_name='Alternativas', index=False)

    print(f"✅ Escalação salva em: {output_file}")
    print("=" * 90 + "\n")

    return lineup


# ========================================================================
# MAIN: PIPELINE MEGA AUTOMÁTICO
# ========================================================================

def main():
    """Pipeline MEGA OPTIMIZER - 100% automático"""

    print("\n" + "=" * 90)
    print("🏆 CARTOLA FC - MEGA OPTIMIZER V2 (ANÁLISE POR POSIÇÃO)")
    print("=" * 90)
    print("📊 Scouts por posição | 🚫 Sem adversários diretos | 🧮 PuLP")
    print("=" * 90)

    # Input do usuário
    budget = float(input("\n💰 Qual seu orçamento (cartoletas)? "))

    print("\n⚽ Formações disponíveis:")
    print("   1 - 4-3-3")
    print("   2 - 3-4-3")
    print("   3 - 3-5-2")
    print("   4 - 4-4-2")
    print("   5 - TODAS (testa todas e escolhe a melhor)")
    formation_choice = input("Escolha (1-5): ").strip()

    formations_list = ['4-3-3', '3-4-3', '3-5-2', '4-4-2']
    if formation_choice in ['1', '2', '3', '4']:
        forced_formation = formations_list[int(formation_choice) - 1]
    else:
        forced_formation = None  # Testa todas

    print("\n" + "=" * 90)
    print(f"💰 Orçamento: C$ {budget:.2f}")
    if forced_formation:
        print(f"⚽ Formação: {forced_formation}")
    else:
        print("⚽ Formação: MELHOR AUTOMÁTICA (testando todas)")
    print("=" * 90)

    # PASSO 1: Carregar dados
    dados_atletas, dados_clubes, dados_partidas = load_all_data()
    if not dados_atletas:
        print("\n❌ ERRO FATAL: Não foi possível carregar dados!")
        return

    # PASSO 2: Processar dados
    df, partidas = process_all_data(dados_atletas, dados_clubes, dados_partidas)

    if len(df) < 12:
        print(f"\n❌ ERRO: Poucos jogadores prováveis ({len(df)})")
        return

    # PASSO 3: Feature Engineering
    df = engineer_features(df)

    # PASSO 4: Otimização
    if forced_formation:
        print(f"\n🧮 ETAPA 4: Otimização (PuLP) | Orçamento: C$ {budget:.2f} | Formação: {forced_formation}")
        lineup = optimize_with_pulp(df, budget, forced_formation)
        formation = forced_formation
        if lineup is not None and len(lineup) == 12:
            print(f"   ⚽ {formation}: Score={lineup['mega_score'].sum():.1f} | Custo=C${lineup['preco'].sum():.1f}")
        else:
            print(f"   ❌ {formation}: Inviável! Tentando todas...")
            lineup, formation = find_best_lineup(df, budget)
    else:
        lineup, formation = find_best_lineup(df, budget)

    if lineup is None or len(lineup) < 12:
        print("\n❌ ERRO: Não foi possível gerar escalação completa!")
        # Tentar com orçamento maior
        print("🔄 Tentando com orçamento ilimitado (C$ 500)...")
        lineup, formation = find_best_lineup(df, 500.0)

        if lineup is None or len(lineup) < 12:
            print("\n❌ ERRO FATAL: Impossível gerar escalação!")
            return

        budget = 500.0

    # PASSO 5: Display e salvar
    display_and_save(lineup, formation, budget, df)

    print("🏆 MEGA OPTIMIZER CONCLUÍDO COM SUCESSO! ⚽\n")


if __name__ == "__main__":
    main()
