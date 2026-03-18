#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
========================================================================
CARTOLA FC - ESCALAÇÃO SMART V2 (Rodada 7)
========================================================================
Script completo com:
  - Busca de atletas da API Cartola (status_id==7 = Provável)
  - Análise de confrontos (favoritos, mando, contexto)
  - Score composto por jogador
  - Justificativa textual por jogador
  - Otimizador PuLP 4-4-2 com orçamento C$113.82
  - Output console com justificativas
  - Export Excel 3 abas
========================================================================
"""

import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import requests

warnings.filterwarnings("ignore")

try:
    from pulp import (
        LpProblem, LpMaximize, LpVariable, LpBinary,
        lpSum, PULP_CBC_CMD, value, LpStatus,
    )
except ImportError:
    sys.exit("PuLP não instalado. Instale com: pip install pulp")

try:
    import openpyxl  # noqa: F401
except ImportError:
    sys.exit("openpyxl não instalado. Instale com: pip install openpyxl")


# ========================================================================
# CONSTANTES
# ========================================================================

API_BASE = "https://api.cartola.globo.com"
ORCAMENTO = 113.82
FORMACAO = "4-4-2"  # 1 GOL + 4 DEF(LAT+ZAG) + 4 MEI + 2 ATA + 1 TEC
MAX_POR_CLUBE = 3

POS_MAP = {1: "GOL", 2: "LAT", 3: "ZAG", 4: "MEI", 5: "ATA", 6: "TEC"}
POS_ORDER = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
STATUS_MAP = {
    2: "Dúvida",
    3: "Suspenso",
    5: "Contundido",
    6: "Nulo",
    7: "Provável",
}

# Confrontos Rodada 7 (pesquisa prévia)
CONFRONTOS_R7 = {
    "PAL": {"favorito": True, "mando": "casa", "adversario": "BOT",
            "contexto": "Palmeiras favorito em casa"},
    "BOT": {"favorito": False, "mando": "fora", "adversario": "PAL",
            "contexto": "Botafogo visitante no Allianz"},
    "BAH": {"favorito": True, "mando": "casa", "adversario": "RBB",
            "contexto": "Bahia favorito em casa"},
    "RBB": {"favorito": False, "mando": "fora", "adversario": "BAH",
            "contexto": "Red Bull Bragantino fora contra Bahia"},
    "CAP": {"favorito": True, "mando": "casa", "adversario": "CRU",
            "contexto": "Athletico-PR favorito; Cruzeiro sem técnico, 5 desfalques"},
    "CRU": {"favorito": False, "mando": "fora", "adversario": "CAP",
            "contexto": "Cruzeiro fragilizado, sem técnico e com desfalques"},
    "MIR": {"favorito": True, "mando": "casa", "adversario": "CFC",
            "contexto": "Mirassol favorito em casa"},
    "CFC": {"favorito": False, "mando": "fora", "adversario": "MIR",
            "contexto": "Coritiba visitante contra Mirassol"},
    "CAM": {"favorito": False, "mando": "casa", "adversario": "SAO",
            "contexto": "Atlético-MG em casa, mas SP líder invicto"},
    "SAO": {"favorito": True, "mando": "fora", "adversario": "CAM",
            "contexto": "São Paulo líder invicto, favorito mesmo fora"},
    "VAS": {"favorito": False, "mando": "casa", "adversario": "FLU",
            "contexto": "Vasco em casa no clássico"},
    "FLU": {"favorito": True, "mando": "fora", "adversario": "VAS",
            "contexto": "Fluminense favorito no clássico carioca"},
    "SAN": {"favorito": True, "mando": "casa", "adversario": "INT",
            "contexto": "Santos invicto em casa; Inter lanterna"},
    "INT": {"favorito": False, "mando": "fora", "adversario": "SAN",
            "contexto": "Inter lanterna, visitante na Vila Belmiro"},
    "GRE": {"favorito": True, "mando": "casa", "adversario": "VIT",
            "contexto": "Grêmio 7 jogos sem perder, favorito em casa"},
    "VIT": {"favorito": False, "mando": "fora", "adversario": "GRE",
            "contexto": "Vitória visitante contra Grêmio embalado"},
    "FLA": {"favorito": True, "mando": "casa", "adversario": "REM",
            "contexto": "Flamengo grande favorito; Remo lanterna"},
    "REM": {"favorito": False, "mando": "fora", "adversario": "FLA",
            "contexto": "Remo lanterna, visitante no Maracanã"},
    "CHA": {"favorito": True, "mando": "casa", "adversario": "COR",
            "contexto": "Chapecoense favorita em casa"},
    "COR": {"favorito": False, "mando": "fora", "adversario": "CHA",
            "contexto": "Corinthians visitante contra Chapecoense"},
}

UNANIMIDADES_R7 = ["Varela", "Arrascaeta", "Pedro"]


# ========================================================================
# 1. FETCH ATLETAS
# ========================================================================

def fetch_atletas():
    """Busca atletas do mercado e retorna DataFrame filtrado."""
    print("📡 Buscando atletas da API Cartola FC...")
    headers = {"User-Agent": "Mozilla/5.0"}

    resp = requests.get(f"{API_BASE}/atletas/mercado", headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    atletas_raw = data.get("atletas", [])
    clubes_api = data.get("clubes", {})

    # Mapa clube_id -> abreviação
    clube_abrev = {}
    for cid, info in clubes_api.items():
        clube_abrev[int(cid)] = info.get("abreviacao", str(cid))

    rows = []
    for a in atletas_raw:
        rows.append({
            "atleta_id": a.get("atleta_id"),
            "nome": a.get("nome", ""),
            "apelido": a.get("apelido", ""),
            "posicao_id": a.get("posicao_id"),
            "clube_id": a.get("clube_id"),
            "status_id": a.get("status_id"),
            "preco": a.get("preco_num", 0.0),
            "media": a.get("media_num", 0.0),
            "jogos": a.get("jogos_num", 0),
            "pontos_ult": a.get("pontos_num", 0.0),
        })

    df = pd.DataFrame(rows)
    df["clube"] = df["clube_id"].map(clube_abrev).fillna("???")
    df["posicao"] = df["posicao_id"].map(POS_MAP).fillna("???")

    total = len(df)

    # Filtrar jogadores de campo: APENAS status_id==7 (Provável) + pelo menos 1 jogo
    df_campo = df[
        (df["posicao_id"].isin([1, 2, 3, 4, 5]))
        & (df["status_id"] == 7)
        & (df["jogos"] >= 1)
    ].copy()

    # TEC (posicao_id==6): todos com preco>0 (status_id não se aplica)
    df_tec = df[(df["posicao_id"] == 6) & (df["preco"] > 0)].copy()

    df_filtered = pd.concat([df_campo, df_tec], ignore_index=True)

    print(f"  ✅ {total} atletas na API → {len(df_campo)} campo prováveis + {len(df_tec)} técnicos = {len(df_filtered)} disponíveis")
    return df_filtered, df, clube_abrev


# ========================================================================
# 2. FETCH PARTIDAS
# ========================================================================

def fetch_partidas(rodada):
    """Busca partidas da rodada."""
    print(f"📡 Buscando partidas da rodada {rodada}...")
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(f"{API_BASE}/partidas/{rodada}", headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  ⚠️ Erro ao buscar partidas: {e}")
        return {}


# ========================================================================
# 3. PESQUISAR FAVORITOS DA RODADA
# ========================================================================

def pesquisar_favoritos_rodada(rodada):
    """Retorna dict com favoritos/mando/contexto dos confrontos."""
    print(f"🔍 Carregando análise de confrontos da rodada {rodada}...")
    print(f"  📊 {len(CONFRONTOS_R7)} clubes mapeados com favoritos/mando/contexto")
    print(f"  🌟 Unanimidades: {', '.join(UNANIMIDADES_R7)}")
    return CONFRONTOS_R7, UNANIMIDADES_R7


# ========================================================================
# 4. CALCULAR SCORE COMPOSTO
# ========================================================================

def calcular_scores(df, confrontos, unanimidades):
    """
    Calcula score composto para cada atleta.
    score = (predicao/max_pred)*0.35 + (media/preco)/max_cb*0.25
            + margem_norm*0.15 + (preco/max_preco)*0.10 + bonus_ctx*0.15
    """
    df = df.copy()

    # Predição simples: média + pontos recentes ponderados
    df["predicao"] = df["media"] * 0.7 + df["pontos_ult"] * 0.3

    # Custo-benefício
    df["custo_beneficio"] = np.where(df["preco"] > 0, df["media"] / df["preco"], 0)

    # MPV = preco * 0.54
    df["mpv"] = df["preco"] * 0.54
    df["margem"] = df["predicao"] - df["mpv"]

    # Bonus contexto
    def calc_bonus(row):
        clube = row["clube"]
        info = confrontos.get(clube, {})
        bonus = 0.0
        if info.get("favorito"):
            bonus += 0.05
        if info.get("mando") == "casa":
            bonus += 0.03
        if row["apelido"] in unanimidades:
            bonus += 0.07
        return bonus

    df["bonus_ctx"] = df.apply(calc_bonus, axis=1)

    # Normalizar componentes
    max_pred = df["predicao"].max() if df["predicao"].max() > 0 else 1
    max_cb = df["custo_beneficio"].max() if df["custo_beneficio"].max() > 0 else 1
    max_preco = df["preco"].max() if df["preco"].max() > 0 else 1
    margem_min = df["margem"].min()
    margem_max = df["margem"].max()
    margem_range = margem_max - margem_min if margem_max > margem_min else 1

    df["score"] = (
        (df["predicao"] / max_pred) * 0.35
        + (df["custo_beneficio"] / max_cb) * 0.25
        + ((df["margem"] - margem_min) / margem_range) * 0.15
        + (df["preco"] / max_preco) * 0.10
        + df["bonus_ctx"] * 0.15 / 0.15  # bonus already 0-0.15 range, normalize to 0-1 for weight
    )
    # Re-normalize: bonus_ctx max ~0.15, so *0.15/0.15 = bonus_ctx itself as contribution
    # Actually let's compute cleanly:
    df["score"] = (
        (df["predicao"] / max_pred) * 0.35
        + (df["custo_beneficio"] / max_cb) * 0.25
        + ((df["margem"] - margem_min) / margem_range) * 0.15
        + (df["preco"] / max_preco) * 0.10
        + df["bonus_ctx"]  # already in 0-0.15 range
    )

    return df


# ========================================================================
# 5. GERAR JUSTIFICATIVA
# ========================================================================

def gerar_justificativa(row, confrontos):
    """Gera texto justificando a escolha de um jogador."""
    parts = []
    clube = row["clube"]
    info = confrontos.get(clube, {})

    # Média e predição
    parts.append(f"Média {row['media']:.2f} pts em {row['jogos']} jogos")

    if row.get("pontos_ult", 0) > 0:
        parts.append(f"fez {row['pontos_ult']:.1f} pts na última rodada")

    # Custo-benefício
    cb = row.get("custo_beneficio", 0)
    if cb > 1.5:
        parts.append(f"excelente custo-benefício ({cb:.2f} pts/C$)")
    elif cb > 1.0:
        parts.append(f"bom custo-benefício ({cb:.2f} pts/C$)")

    # Contexto do confronto
    if info:
        adv = info.get("adversario", "?")
        if info.get("favorito"):
            parts.append(f"time favorito vs {adv}")
        if info.get("mando") == "casa":
            parts.append("jogando em casa")
        elif info.get("mando") == "fora":
            parts.append("jogando fora")
        ctx = info.get("contexto", "")
        if ctx:
            parts.append(ctx)

    # Unanimidade
    if row["apelido"] in UNANIMIDADES_R7:
        parts.append("UNANIMIDADE da rodada")

    # MPV
    mpv = row.get("mpv", 0)
    pred = row.get("predicao", 0)
    if pred > mpv and mpv > 0:
        parts.append(f"predição ({pred:.2f}) acima do MPV ({mpv:.2f}) → tende a valorizar")

    return ". ".join(parts) + "."


# ========================================================================
# 6. OTIMIZAR 4-4-2 COM PuLP
# ========================================================================

def otimizar_442(df, orcamento):
    """
    Otimização PuLP para formação 4-4-2.
    1 GOL + 4 DEF (LAT+ZAG, min 1 de cada) + 4 MEI + 2 ATA + 1 TEC = 12
    """
    print(f"\n🧮 Otimizando formação {FORMACAO} com orçamento C${orcamento:.2f}...")

    # Separar por posição
    gol = df[df["posicao_id"] == 1].reset_index(drop=True)
    lat = df[df["posicao_id"] == 2].reset_index(drop=True)
    zag = df[df["posicao_id"] == 3].reset_index(drop=True)
    mei = df[df["posicao_id"] == 4].reset_index(drop=True)
    ata = df[df["posicao_id"] == 5].reset_index(drop=True)
    tec = df[df["posicao_id"] == 6].reset_index(drop=True)

    print(f"  Disponíveis: GOL={len(gol)} LAT={len(lat)} ZAG={len(zag)} MEI={len(mei)} ATA={len(ata)} TEC={len(tec)}")

    if len(gol) < 1 or len(lat) < 1 or len(zag) < 1 or len(mei) < 4 or len(ata) < 2 or len(tec) < 1:
        print("  ❌ Jogadores insuficientes para formação 4-4-2!")
        return None

    # Criar problema
    prob = LpProblem("Cartola_442", LpMaximize)

    # Variáveis binárias para cada jogador
    all_players = pd.concat([gol, lat, zag, mei, ata, tec], ignore_index=True)
    n = len(all_players)
    x = [LpVariable(f"x_{i}", cat=LpBinary) for i in range(n)]

    # Preços e scores
    precos = all_players["preco"].values
    scores = all_players["score"].values
    pos_ids = all_players["posicao_id"].values
    clube_ids = all_players["clube_id"].values

    # Objetivo: maximizar score total
    prob += lpSum(scores[i] * x[i] for i in range(n))

    # Restrição de orçamento: custo <= orcamento
    prob += lpSum(precos[i] * x[i] for i in range(n)) <= orcamento

    # Restrição de uso mínimo do orçamento: custo >= 97% do orçamento
    prob += lpSum(precos[i] * x[i] for i in range(n)) >= orcamento * 0.97

    # Restrições de posição
    idx_gol = [i for i in range(n) if pos_ids[i] == 1]
    idx_lat = [i for i in range(n) if pos_ids[i] == 2]
    idx_zag = [i for i in range(n) if pos_ids[i] == 3]
    idx_mei = [i for i in range(n) if pos_ids[i] == 4]
    idx_ata = [i for i in range(n) if pos_ids[i] == 5]
    idx_tec = [i for i in range(n) if pos_ids[i] == 6]

    # Exatamente: 1 GOL, 4 MEI, 2 ATA, 1 TEC
    prob += lpSum(x[i] for i in idx_gol) == 1
    prob += lpSum(x[i] for i in idx_mei) == 4
    prob += lpSum(x[i] for i in idx_ata) == 2
    prob += lpSum(x[i] for i in idx_tec) == 1

    # DEF: LAT + ZAG == 4, cada um >= 1
    prob += lpSum(x[i] for i in idx_lat) + lpSum(x[i] for i in idx_zag) == 4
    prob += lpSum(x[i] for i in idx_lat) >= 1
    prob += lpSum(x[i] for i in idx_zag) >= 1

    # Max 3 por clube
    clubes_unicos = all_players["clube_id"].unique()
    for c in clubes_unicos:
        idx_clube = [i for i in range(n) if clube_ids[i] == c]
        prob += lpSum(x[i] for i in idx_clube) <= MAX_POR_CLUBE

    # Resolver
    solver = PULP_CBC_CMD(msg=0, timeLimit=30)
    prob.solve(solver)

    status = LpStatus[prob.status]
    if status != "Optimal":
        print(f"  ⚠️ Status do solver: {status}")
        # Relax budget constraint and retry
        print("  🔄 Tentando relaxar restrição de orçamento mínimo...")
        prob2 = LpProblem("Cartola_442_relaxed", LpMaximize)
        x2 = [LpVariable(f"x2_{i}", cat=LpBinary) for i in range(n)]
        prob2 += lpSum(scores[i] * x2[i] for i in range(n))
        prob2 += lpSum(precos[i] * x2[i] for i in range(n)) <= orcamento
        prob2 += lpSum(precos[i] * x2[i] for i in range(n)) >= orcamento * 0.90
        prob2 += lpSum(x2[i] for i in idx_gol) == 1
        prob2 += lpSum(x2[i] for i in idx_mei) == 4
        prob2 += lpSum(x2[i] for i in idx_ata) == 2
        prob2 += lpSum(x2[i] for i in idx_tec) == 1
        prob2 += lpSum(x2[i] for i in idx_lat) + lpSum(x2[i] for i in idx_zag) == 4
        prob2 += lpSum(x2[i] for i in idx_lat) >= 1
        prob2 += lpSum(x2[i] for i in idx_zag) >= 1
        for c in clubes_unicos:
            idx_clube = [i for i in range(n) if clube_ids[i] == c]
            prob2 += lpSum(x2[i] for i in idx_clube) <= MAX_POR_CLUBE
        prob2.solve(solver)
        status2 = LpStatus[prob2.status]
        if status2 != "Optimal":
            print(f"  ❌ Solver falhou mesmo relaxado: {status2}")
            return None
        x = x2

    # Extrair escalação
    selected = [i for i in range(n) if value(x[i]) == 1]
    escalacao = all_players.iloc[selected].copy()
    custo_total = escalacao["preco"].sum()

    print(f"  ✅ Solver: {status} | {len(selected)} jogadores | C${custo_total:.2f} ({custo_total/orcamento*100:.1f}%)")
    return escalacao


# ========================================================================
# 7. EXPORTAR EXCEL 3 ABAS
# ========================================================================

def exportar_excel(escalacao, confrontos, df_todos, rodada):
    """Exporta Excel com 3 abas: Escalação, Análise Confrontos, Top Prováveis."""
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = output_dir / f"escalacao_smart_v2_r{rodada}_{ts}.xlsx"

    # Aba 1: Escalação
    esc_cols = ["posicao_id", "posicao", "apelido", "clube", "preco", "media", "jogos",
                "pontos_ult", "predicao", "mpv", "score", "justificativa"]
    esc_df = escalacao[[c for c in esc_cols if c in escalacao.columns]].copy()
    esc_df = esc_df.sort_values("posicao_id" if "posicao_id" in esc_df.columns else "posicao")
    # Drop posicao_id from final output (internal use only)
    if "posicao_id" in esc_df.columns:
        esc_df = esc_df.drop(columns=["posicao_id"])

    # Aba 2: Análise Confrontos
    conf_rows = []
    seen = set()
    for abrev, info in confrontos.items():
        adv = info.get("adversario", "?")
        par = tuple(sorted([abrev, adv]))
        if par in seen:
            continue
        seen.add(par)
        mand = abrev if info.get("mando") == "casa" else adv
        visit = adv if info.get("mando") == "casa" else abrev
        fav = abrev if info.get("favorito") else adv
        ctx_home = confrontos.get(mand, {}).get("contexto", "")
        ctx_away = confrontos.get(visit, {}).get("contexto", "")
        conf_rows.append({
            "Mandante": mand,
            "Visitante": visit,
            "Favorito": fav,
            "Contexto Mandante": ctx_home,
            "Contexto Visitante": ctx_away,
        })
    conf_df = pd.DataFrame(conf_rows)

    # Aba 3: Top Prováveis por posição
    top_rows = []
    for pos_id in [1, 2, 3, 4, 5, 6]:
        pos_df = df_todos[df_todos["posicao_id"] == pos_id].nlargest(10, "score")
        for _, row in pos_df.iterrows():
            top_rows.append({
                "Posição": POS_MAP.get(pos_id, "?"),
                "Jogador": row["apelido"],
                "Clube": row["clube"],
                "Preço": row["preco"],
                "Média": row["media"],
                "Score": round(row["score"], 4),
                "Predição": round(row.get("predicao", 0), 2),
            })
    top_df = pd.DataFrame(top_rows)

    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        esc_df.to_excel(writer, sheet_name="Escalação", index=False)
        conf_df.to_excel(writer, sheet_name="Análise Confrontos", index=False)
        top_df.to_excel(writer, sheet_name="Top Prováveis", index=False)

    print(f"\n💾 Excel salvo em: {filepath}")
    return filepath


# ========================================================================
# 8. MAIN
# ========================================================================

def main():
    """Orquestra todo o pipeline."""
    print("=" * 50)
    print("  CARTOLA FC - ESCALAÇÃO SMART V2")
    print(f"  Orçamento: C${ORCAMENTO} | Formação: {FORMACAO}")
    print("=" * 50)

    # 1. Buscar mercado para saber a rodada
    print("\n📡 Buscando status do mercado...")
    try:
        mercado = requests.get(
            f"{API_BASE}/mercado/status",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=30,
        ).json()
        rodada = mercado.get("rodada_atual", 7)
        status_mercado = mercado.get("status_mercado")
        print(f"  Rodada atual: {rodada} | Status mercado: {status_mercado}")
    except Exception as e:
        print(f"  ⚠️ Erro ao buscar mercado: {e}. Usando rodada 7.")
        rodada = 7

    # 2. Buscar atletas
    df_filtered, df_all, clube_abrev = fetch_atletas()

    # 3. Buscar partidas
    fetch_partidas(rodada)

    # 4. Pesquisar favoritos
    confrontos, unanimidades = pesquisar_favoritos_rodada(rodada)

    # 5. Calcular scores
    print("\n📊 Calculando scores compostos...")
    df_scored = calcular_scores(df_filtered, confrontos, unanimidades)
    print(f"  ✅ Scores calculados para {len(df_scored)} atletas")

    # 6. Otimizar
    escalacao = otimizar_442(df_scored, ORCAMENTO)
    if escalacao is None or len(escalacao) == 0:
        print("\n❌ Não foi possível montar a escalação. Abortando.")
        return

    # 7. Gerar justificativas
    print("\n📝 Gerando justificativas...")
    escalacao["justificativa"] = escalacao.apply(
        lambda row: gerar_justificativa(row, confrontos), axis=1
    )

    # 8. Ordenar por posição
    escalacao = escalacao.sort_values(
        "posicao_id", key=lambda s: s.map(POS_ORDER)
    ).reset_index(drop=True)

    # 9. Output console
    custo_total = escalacao["preco"].sum()
    pct_orc = custo_total / ORCAMENTO * 100

    print("\n" + "=" * 60)
    print("  ESCALAÇÃO RODADA", rodada)
    print(f"  Orçamento: C${ORCAMENTO} | Formação: {FORMACAO}")
    print("=" * 60)

    for _, row in escalacao.iterrows():
        pos = POS_MAP.get(row["posicao_id"], "?")
        print(f"\n[{pos}] {row['apelido']} ({row['clube']}) - C${row['preco']:.2f} | Média: {row['media']:.2f}")
        print(f"  ✅ PROVÁVEL | 🏆 Score: {row['score']:.4f}")
        print(f"  💡 Justificativa: {row['justificativa']}")

    # Capitão: maior score entre jogadores de campo (não TEC)
    campo = escalacao[escalacao["posicao_id"] != 6]
    if len(campo) > 0:
        capitao = campo.loc[campo["score"].idxmax()]
        print(f"\n👑 CAPITÃO: {capitao['apelido']} ({capitao['clube']}) — média {capitao['media']:.2f} — maior score ({capitao['score']:.4f})")

    mpv_medio = escalacao[escalacao["posicao_id"] != 6]["preco"].mean()
    print(f"\n💰 CUSTO TOTAL: C${custo_total:.2f} ({pct_orc:.1f}% do orçamento)")
    print(f"📊 Preço médio (campo): C${mpv_medio:.2f}/jogador")

    # 10. Unanimidades na escalação?
    unani_esc = [r["apelido"] for _, r in escalacao.iterrows() if r["apelido"] in UNANIMIDADES_R7]
    if unani_esc:
        print(f"🌟 Unanimidades escaladas: {', '.join(unani_esc)}")
    else:
        print(f"⚠️ Nenhuma unanimidade escalada (disputa por custo-benefício)")

    # 11. Exportar Excel
    print("\n📊 Exportando Excel...")
    exportar_excel(escalacao, confrontos, df_scored, rodada)

    print("\n✅ Pipeline concluído!")


if __name__ == "__main__":
    main()
