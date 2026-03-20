#!/usr/bin/env python3
"""
Sistema Completo de Análise e Predição do Cartola FC

Uso simples (roda TUDO automaticamente):
    python main.py

Opções avançadas (opcionais):
    python main.py --orcamento 110 --formacao 4-3-3
    python main.py --modo valorizar
    python main.py --reserva-luxo
    python main.py --v3-features
    python main.py --janela-historico 7

Pipeline obrigatório (executado sempre, sem precisar de flags):
  1.  Inicializar API
  2.  Coletar atletas + partidas da rodada
  3.  Carregar histórico de pontuações + H2H confrontos
  4.  Classificar confrontos (Grupo A/B/C) via últimos N H2H
  5.  Desfalques + análise de confrontos
  6.  Feature Engineering
  7.  Treinar modelo ML (ou heurística se histórico insuficiente)
  8.  Gerar predições cruzadas: ML × matchup × H2H × forma recente
  9.  Specialist Checklist (capitão, avisos, validação)
  10. Otimização genética → escalacão final única
  [Opcional] Reserva de Luxo
"""

import argparse
import logging
import os
import sqlite3
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import yaml
from dotenv import load_dotenv

load_dotenv()

from src.api.client import CartolaAPIClient
from src.data.collector import CartolaDataCollector
from src.ml.features import FeatureEngineer
from src.ml.predictor import CartolaPredictor
from src.ml.optimizer import GeneticTeamOptimizer
from src.optimizer.factory import CartolaOptimizer
from src.features.odds_integrator import OddsIntegrator
from src.utils.validators import validar_mercado, validar_formacao, filtrar_atletas_com_jogo

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURAÇÃO E ARGS
# ============================================================

def carregar_config(config_path: str = "config.yaml") -> dict:
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"⚠️  {config_path} não encontrado, usando defaults")
        return {}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cartola FC – Roda python main.py e gera a melhor escalação automaticamente"
    )
    p.add_argument("--orcamento",         type=float, default=float(os.getenv("CARTOLA_ORCAMENTO", 110.0)))
    p.add_argument("--formacao",          type=str,   default=os.getenv("CARTOLA_FORMACAO", ""))
    p.add_argument("--modo",              type=str,   default=os.getenv("CARTOLA_MODO", "equilibrado"),
                   choices=["equilibrado", "pontuar", "valorizar"])
    p.add_argument("--output-dir",        type=str,   default="output")
    p.add_argument("--reserva-luxo",      action="store_true", default=False)
    p.add_argument("--v3-features",       action="store_true", default=False)
    p.add_argument("--janela-historico",  type=int,   default=5,
                   help="Número de rodadas/confrontos no histórico H2H (padrão: 5)")
    return p.parse_args()


# ============================================================
# HISTÓRICO DE CONFRONTOS H2H
# ============================================================

def carregar_historico_confrontos(conn: sqlite3.Connection, rodada_atual: int) -> pd.DataFrame:
    """
    Busca todos os resultados históricos de partidas anteriores à rodada atual.
    Usa clube_id_a / clube_id_b / placar_a / placar_b (colunas novas do collector).
    """
    query = """
        SELECT
            clube_id_a   AS clube_a,
            clube_id_b   AS clube_b,
            rodada,
            placar_a     AS gols_a,
            placar_b     AS gols_b,
            CASE
                WHEN placar_a > placar_b  THEN 'vitoria_a'
                WHEN placar_a < placar_b  THEN 'vitoria_b'
                ELSE 'empate'
            END AS resultado
        FROM partidas
        WHERE rodada < ?
          AND placar_a IS NOT NULL
          AND placar_b IS NOT NULL
        ORDER BY rodada DESC
    """
    try:
        return pd.read_sql_query(query, conn, params=(rodada_atual,))
    except Exception:
        logger.warning("⚠️  Histórico H2H não disponível ainda (rodadas futuras ou banco novo)")
        return pd.DataFrame(columns=["clube_a", "clube_b", "rodada", "gols_a", "gols_b", "resultado"])


def ultimos_n_confrontos(hist_df: pd.DataFrame, ca: int, cb: int, n: int) -> pd.DataFrame:
    """Filtra os últimos N jogos diretos entre dois clubes (ambos os sentidos)."""
    mask = (
        ((hist_df["clube_a"] == ca) & (hist_df["clube_b"] == cb)) |
        ((hist_df["clube_a"] == cb) & (hist_df["clube_b"] == ca))
    )
    return hist_df[mask].head(n)


def score_h2h(h2h_df: pd.DataFrame, clube_id: int) -> float:
    """Aproveitamento do clube nos H2H: 0.0 (perdeu tudo) a 1.0 (ganhou tudo)."""
    if h2h_df.empty:
        return 0.5
    pts = 0.0
    for _, row in h2h_df.iterrows():
        if row["clube_a"] == clube_id and row["resultado"] == "vitoria_a":
            pts += 1
        elif row["clube_b"] == clube_id and row["resultado"] == "vitoria_b":
            pts += 1
        elif row["resultado"] == "empate":
            pts += 0.5
    return pts / len(h2h_df)


# ============================================================
# CLASSIFICAÇÃO DE CONFRONTOS (Grupo A/B/C)
# ============================================================

def classificar_confrontos(
    partidas_rodada: pd.DataFrame,
    hist_df: pd.DataFrame,
    janela: int
) -> pd.DataFrame:
    """
    Enriquece partidas da rodada com:
      tipo_confronto  — 'A' (favorito), 'B' (aberto), 'C' (truncado)
      media_gols_h2h  — média de gols nos últimos N H2H
      score_h2h_a     — aproveitamento do time casa
      score_h2h_b     — aproveitamento do time visitante
    """
    if partidas_rodada.empty:
        return partidas_rodada

    tipos, media_gols, sc_a_list, sc_b_list = [], [], [], []

    for _, row in partidas_rodada.iterrows():
        ca = int(row.get("clube_id_a", row.get("clube_casa_id", 0)))
        cb = int(row.get("clube_id_b", row.get("clube_visitante_id", 0)))

        h2h = ultimos_n_confrontos(hist_df, ca, cb, janela)

        mg = (
            (h2h["gols_a"].fillna(0) + h2h["gols_b"].fillna(0)).mean()
            if not h2h.empty and "gols_a" in h2h.columns
            else 2.5
        )

        sc_a = score_h2h(h2h, ca)
        sc_b = score_h2h(h2h, cb)
        diff = abs(sc_a - sc_b)

        if diff >= 0.3:
            tipo = "A"  # Amplo favoritismo
        elif mg < 1.5:
            tipo = "C"  # Jogo truncado / 0x0
        else:
            tipo = "B"  # Equilibrado

        tipos.append(tipo)
        media_gols.append(round(mg, 2))
        sc_a_list.append(round(sc_a, 3))
        sc_b_list.append(round(sc_b, 3))

    partidas_rodada = partidas_rodada.copy()
    partidas_rodada["tipo_confronto"] = tipos
    partidas_rodada["media_gols_h2h"] = media_gols
    partidas_rodada["score_h2h_a"]    = sc_a_list
    partidas_rodada["score_h2h_b"]    = sc_b_list

    logger.info(
        f"📊 Confrontos da rodada — "
        f"Grupo A: {tipos.count('A')} | B: {tipos.count('B')} | C: {tipos.count('C')}"
    )
    return partidas_rodada


# ============================================================
# CRUZAMENTO DE SCORES (ML + H2H + FORMA)
# ============================================================

def cruzar_scores(
    predicoes_df: pd.DataFrame,
    atletas_df: pd.DataFrame,
    partidas_enriquecidas: pd.DataFrame,
    hist_df: pd.DataFrame,
    janela: int
) -> pd.DataFrame:
    """
    Gera coluna `score_cruzado`:
        score_cruzado = score_ML × mult_confronto × norm_h2h

    Multiplicadores:
        Grupo A = 1.15  (favorito claro)
        Grupo B = 1.00  (equilibrado)
        Grupo C = 0.85  (truncado)

    norm_h2h = 0.7 + aproveitamento_h2h × 0.6   (escala 0.7 – 1.3)
    """
    if predicoes_df.empty:
        return predicoes_df

    # Mapear clube → tipo e score H2H
    clube_tipo: Dict[int, str]   = {}
    clube_h2h:  Dict[int, float] = {}

    if not partidas_enriquecidas.empty:
        for _, row in partidas_enriquecidas.iterrows():
            ca = int(row.get("clube_id_a", row.get("clube_casa_id", 0)))
            cb = int(row.get("clube_id_b", row.get("clube_visitante_id", 0)))
            tipo = row.get("tipo_confronto", "B")
            clube_tipo[ca] = tipo
            clube_tipo[cb] = tipo
            clube_h2h[ca]  = row.get("score_h2h_a", 0.5)
            clube_h2h[cb]  = row.get("score_h2h_b", 0.5)

    pred = predicoes_df.copy()

    # Garantir clube_id no df de predições
    if "clube_id" not in pred.columns and "atleta_id" in pred.columns and "clube_id" in atletas_df.columns:
        mapa = atletas_df.set_index("atleta_id")["clube_id"].to_dict()
        pred["clube_id"] = pred["atleta_id"].map(mapa).fillna(0).astype(int)

    pred["tipo_confronto"] = pred.get("clube_id", pd.Series(0, index=pred.index)).map(
        lambda cid: clube_tipo.get(int(cid), "B")
    )
    pred["score_h2h"] = pred.get("clube_id", pd.Series(0, index=pred.index)).map(
        lambda cid: clube_h2h.get(int(cid), 0.5)
    )

    mult_map = {"A": 1.15, "B": 1.0, "C": 0.85}
    pred["mult_confronto"] = pred["tipo_confronto"].map(mult_map).fillna(1.0)

    score_base_col = next(
        (c for c in ["score_final", "predicao_ajustada", "predicao"] if c in pred.columns),
        None
    )
    if score_base_col is None:
        pred["score_cruzado"] = 0.0
        return pred

    norm_h2h = 0.7 + pred["score_h2h"] * 0.6
    pred["score_cruzado"] = pred[score_base_col] * pred["mult_confronto"] * norm_h2h

    logger.info(
        f"🔀 score_cruzado gerado para {len(pred)} atletas "
        f"(últimos {janela} confrontos H2H)"
    )
    return pred


# ============================================================
# HELPER DE IMPRESSÃO
# ============================================================

def imprimir_escalacao_final(team_df: pd.DataFrame, stats: dict, specialist_result: dict, patrimonio: float):
    """Imprime escalação formatada no terminal, uma única vez."""
    sep = "=" * 70
    print(f"\n{sep}")
    print("          🏆  ESCALAÇÃO FINAL  🏆")
    print(sep)

    # Agrupa por posição
    for grupo, label in [
        (["gol"],                       "🧄 GOLEIRO"),
        (["zagueiro", "lateral"],        "🛡️  DEFESA"),
        (["volante", "meia", "ponta"],   "🎯 MEIO-CAMPO"),
        (["atacante"],                   "⚽  ATAQUE"),
    ]:
        pos_col = next((c for c in ["posicao", "posicao_id", "pos"] if c in team_df.columns), None)
        if pos_col:
            subset = team_df[team_df[pos_col].isin(grupo)]
        else:
            subset = team_df  # fallback: imprime tudo junto

        if not subset.empty:
            print(f"\n{label}")
            for _, r in subset.iterrows():
                apelido  = r.get("apelido", r.get("nome", "?"))
                preco    = r.get("preco",   0.0)
                score    = r.get("score_cruzado", r.get("score_final", 0.0))
                tipo_c   = r.get("tipo_confronto", "-")
                h2h_val  = r.get("score_h2h", 0.0)
                selo     = r.get("selo_valorizacao", "")
                print(
                    f"   {apelido:<22} "
                    f"C${preco:>5.2f}  "
                    f"Score:{score:>6.2f}  "
                    f"Conf:{tipo_c}  "
                    f"H2H:{h2h_val:.2f}  "
                    f"{selo}"
                )

    print(f"\n{'─'*70}")
    print(f"  ⚽ Pontos Preditos : {stats.get('total_pontos_preditos', 0):.2f}")
    print(f"  💰 Preço Total      : C${stats.get('total_preco', 0):.2f} / C${patrimonio:.2f}")
    cap = specialist_result.get('recommended_captain', '')
    if cap:
        print(f"  ⭐ Capitão          : {cap}")
    print(sep + "\n")


def salvar_xlsx(team_df: pd.DataFrame, output_dir: Path, rodada: int, sufixo: str = "") -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    nome = f"escalacao_r{rodada}_{sufixo}_{ts}.xlsx" if sufixo else f"escalacao_r{rodada}_{ts}.xlsx"
    path = output_dir / nome
    team_df.to_excel(path, index=False)
    logger.info(f"💾 Escalação salva: {path}")
    return path


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

def main():
    args   = parse_args()
    config = carregar_config()

    print()
    print("=" * 70)
    print("  🏆  CARTOLA FC – PIPELINE COMPLETO  |  python main.py")
    print(f"  Orçamento: C${args.orcamento:.0f}  |  Modo: {args.modo.upper()}  |  H2H janela: {args.janela_historico}")
    print("=" * 70)

    EMAIL     = os.getenv('CARTOLA_EMAIL')
    PASSWORD  = os.getenv('CARTOLA_PASSWORD')
    PATRIMONIO = args.orcamento
    FORMACAO   = args.formacao if args.formacao else None
    JANELA     = args.janela_historico

    if FORMACAO and not validar_formacao(FORMACAO):
        logger.error(f"❌ Formação '{FORMACAO}' inválida.")
        return

    # ----------------------------------------------------------
    # [1/10] API
    # ----------------------------------------------------------
    logger.info("📡 [1/10] Inicializando cliente API...")
    api_client = CartolaAPIClient(email=EMAIL, password=PASSWORD)

    # ----------------------------------------------------------
    # [2/10] COLETA DE DADOS
    # ----------------------------------------------------------
    logger.info("📥 [2/10] Coletando dados da rodada...")
    collector = CartolaDataCollector(api_client)

    try:
        mercado      = collector.collect_mercado_status()
        mercado_info = validar_mercado(mercado)
        if not mercado_info['valido']:
            logger.error(f"❌ {mercado_info['mensagem']}")
            return

        rodada_atual = mercado_info['rodada_atual']
        logger.info(f"📊 {mercado_info['mensagem']}")

        atletas_df = collector.collect_atletas_mercado(rodada_atual)
        if atletas_df.empty:
            logger.error("❌ Nenhum atleta encontrado.")
            return

        collector.collect_partidas(rodada_atual)
        logger.info(f"✅ {len(atletas_df)} atletas coletados")

    except Exception as e:
        logger.error(f"❌ Erro na coleta: {e}")
        return

    # ----------------------------------------------------------
    # [3/10] HISTÓRICO + H2H
    # ----------------------------------------------------------
    logger.info("📚 [3/10] Carregando histórico + H2H...")
    conn = sqlite3.connect(collector.db_path)

    historico_df = pd.read_sql_query("""
        SELECT p.*, a.clube_id FROM pontuacoes p
        JOIN atletas a ON p.atleta_id = a.atleta_id
        WHERE p.rodada >= ? AND p.rodada <= ?
        ORDER BY p.atleta_id, p.rodada
    """, conn, params=(max(1, rodada_atual - 15), rodada_atual))

    partidas_df = pd.read_sql_query("""
        SELECT * FROM partidas WHERE rodada >= ? AND rodada <= ?
    """, conn, params=(max(1, rodada_atual - 15), rodada_atual))

    hist_h2h_df = carregar_historico_confrontos(conn, rodada_atual)
    conn.close()

    logger.info(
        f"📊 Histórico: {len(historico_df)} reg | Partidas: {len(partidas_df)} | H2H base: {len(hist_h2h_df)}"
    )

    # Clubes
    try:
        r = requests.get(
            "https://api.cartola.globo.com/clubes",
            headers={'User-Agent': 'Mozilla/5.0'}, timeout=10
        )
        clubes_raw = r.json()
        clubes_df  = pd.DataFrame([
            {'id': int(k), 'nome': v.get('nome',''), 'abreviacao': v.get('abreviacao','')}
            for k, v in clubes_raw.items()
        ])
        logger.info(f"✅ {len(clubes_df)} clubes carregados")
    except Exception:
        clubes_df = atletas_df[['clube_id']].drop_duplicates().rename(columns={'clube_id':'id'})
        clubes_df['nome'] = clubes_df['id'].astype(str)
        logger.warning("⚠️  Clubes via fallback")

    atletas_df = filtrar_atletas_com_jogo(atletas_df, rodada_atual, partidas_df)
    logger.info(f"✅ {len(atletas_df)} atletas com jogo confirmado")

    # ----------------------------------------------------------
    # [4/10] ANÁLISE DE CONFRONTOS (Grupo A/B/C)
    # ----------------------------------------------------------
    logger.info("🔍 [4/10] Classificando confrontos da rodada...")
    partidas_rodada = (
        partidas_df[partidas_df['rodada'] == rodada_atual].copy()
        if not partidas_df.empty else pd.DataFrame()
    )
    partidas_enriquecidas = classificar_confrontos(partidas_rodada, hist_h2h_df, JANELA)

    if not partidas_enriquecidas.empty:
        cols_exibir = [c for c in
            ["clube_id_a","clube_id_b","tipo_confronto","media_gols_h2h","score_h2h_a","score_h2h_b"]
            if c in partidas_enriquecidas.columns]
        logger.info(f"\n{partidas_enriquecidas[cols_exibir].to_string(index=False)}")

    # ----------------------------------------------------------
    # [5/10] FEATURE ENGINEERING
    # ----------------------------------------------------------
    logger.info("🔧 [5/10] Gerando features...")
    if len(historico_df) > 0:
        if args.v3_features:
            historico_df = FeatureEngineer.engineer_all_features_v3(historico_df, partidas_df)
            logger.info("✅ Feature Engineering V3")
        else:
            historico_df = FeatureEngineer.engineer_all_features(historico_df, partidas_df)
            logger.info("✅ Feature Engineering V2")
    else:
        logger.warning("⚠️  Sem histórico para feature engineering")

    # ----------------------------------------------------------
    # [6/10] ODDS (sempre tenta, sem exigir flag)
    # ----------------------------------------------------------
    logger.info("🎲 [6/10] Integrando odds...")
    try:
        odds_integrator = OddsIntegrator()
        atletas_df = odds_integrator.enrich(atletas_df, partidas_df, clubes_df)
        logger.info("✅ Odds integradas")
    except Exception as e:
        logger.warning(f"⚠️  Odds não disponíveis ({e}) — continuando sem odds")

    # ----------------------------------------------------------
    # [7/10] TREINO DO MODELO ML
    # ----------------------------------------------------------
    logger.info("🧠 [7/10] Treinando modelo ML...")
    predictor   = CartolaPredictor(model_type=config.get('ml',{}).get('model_type','rf'))
    dados_treino = (
        historico_df[historico_df['rodada'] < rodada_atual]
        if len(historico_df) > 0 else pd.DataFrame()
    )
    can_train = len(dados_treino) >= CartolaPredictor.HISTORICO_MINIMO

    if can_train:
        metrics = predictor.train(dados_treino, validate=True)
        model_path = Path("data/models/cartola_predictor.pkl")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        predictor.save_model(str(model_path))
        logger.info(f"✅ Modelo treinado | {metrics}")
    else:
        logger.warning(
            f"⚠️  Histórico insuficiente ({len(dados_treino)} reg) — usando heurística"
        )

    # ----------------------------------------------------------
    # [8/10] PREDIÇÕES + CRUZAMENTO H2H
    # ----------------------------------------------------------
    logger.info(f"🎯 [8/10] Gerando predições cruzadas (ML + H2H últimos {JANELA})...")

    if can_train and predictor.is_trained:
        ultimos = (
            historico_df.sort_values('rodada')
            .groupby('atleta_id').tail(JANELA)
        )
        ultimos_agg = (
            ultimos.sort_values('rodada')
            .groupby('atleta_id').last()
            .reset_index()
        )
        dados_pred = atletas_df.merge(
            ultimos_agg, on='atleta_id', how='left', suffixes=('','_hist')
        ).fillna(0)
        predicoes_df = predictor.predict_full(dados_pred)
        logger.info("✅ Predições com pesos táticos geradas")
    else:
        predicoes_df = CartolaPredictor.fallback_heuristica(atletas_df)
        predicoes_df['predicao_ajustada'] = predicoes_df['predicao']
        predicoes_df = CartolaPredictor.add_valorizacao_flag(predicoes_df)

    # Score por modo
    score_col = (
        'predicao_ajustada' if 'predicao_ajustada' in predicoes_df.columns else 'predicao'
    )
    if args.modo == 'valorizar' and 'valorizacao_score' in predicoes_df.columns:
        predicoes_df['score_final'] = (
            predicoes_df[score_col] * 0.4 + predicoes_df['valorizacao_score'] * 0.6
        )
    elif args.modo == 'pontuar':
        predicoes_df['score_final'] = predicoes_df[score_col]
    else:
        val_w = predicoes_df.get('valorizacao_score', pd.Series(50, index=predicoes_df.index))
        predicoes_df['score_final'] = predicoes_df[score_col] * 0.65 + val_w * 0.35

    logger.info(f"⚖️  Modo {args.modo.upper()} aplicado ao score")

    # Cruzamento H2H
    predicoes_df = cruzar_scores(
        predicoes_df, atletas_df, partidas_enriquecidas, hist_h2h_df, JANELA
    )

    ranking_col = 'score_cruzado' if 'score_cruzado' in predicoes_df.columns else 'score_final'
    predicoes_df['fitness_score'] = predicoes_df[ranking_col]

    top10 = predicoes_df.nlargest(10, ranking_col)
    cols_top = [c for c in ['apelido','posicao_id','preco',ranking_col,'tipo_confronto','score_h2h']
                if c in top10.columns]
    logger.info(f"\n🏆 Top 10 pré-otimização:\n{top10[cols_top].to_string(index=False)}")

    # ----------------------------------------------------------
    # [9/10] SPECIALIST CHECKLIST
    # ----------------------------------------------------------
    logger.info("🔎 [9/10] Executando Specialist Checklist...")
    specialist_result: dict = {}
    try:
        from src.ml.specialist_logic import CartolaPrescalingChecklist
        checklist = CartolaPrescalingChecklist(
            rodada=rodada_atual,
            budget=PATRIMONIO,
            atletas_df=atletas_df,
            partidas_df=(
                partidas_enriquecidas if not partidas_enriquecidas.empty
                else partidas_df
            ),
            predicoes_df=predicoes_df,
            modo=args.modo,
        )
        specialist_result = checklist.run()
        logger.info(
            f"✅ Checklist OK | Capitão sugerido: "
            f"{specialist_result.get('recommended_captain', 'N/A')}"
        )
        for w in specialist_result.get('warnings', []):
            logger.warning(f"   ⚠️  {w}")
    except Exception as e:
        logger.error(
            f"❌ Specialist falhou: {e}\n{traceback.format_exc()}"
        )
        # Não interrompe — prossegue com otimização sem checklist

    # ----------------------------------------------------------
    # [10/10] OTIMIZAÇÃO GENÉTICA → ESCALAÇÃO Única
    # ----------------------------------------------------------
    logger.info("🧬 [10/10] Otimizando time...")
    opt_config = config.get('optimizer', {})

    pred_opt = predicoes_df.copy()
    if 'predicao_std' not in pred_opt.columns:
        pred_opt['predicao_std'] = 0.0

    if 'odds_score' in pred_opt.columns:
        pred_opt['fitness_score'] = (
            pred_opt['fitness_score'] * 0.8
            + pred_opt['odds_score'] * 20 * 0.2
        )

    try:
        optimizer = GeneticTeamOptimizer(
            atletas_df=atletas_df,
            predicoes=pred_opt,
            patrimonio=PATRIMONIO,
            formacao=FORMACAO or '4-3-3',
            population_size=opt_config.get('population_size', 250),
            generations=opt_config.get('generations', 150),
            mutation_rate=opt_config.get('mutation_rate', 0.20),
            elite_size=opt_config.get('elite_size', 20),
            max_mesmo_clube=opt_config.get('max_mesmo_clube', 3),
            penalidade_variancia=opt_config.get('penalidade_variancia', True),
        )
        best_team, stats = optimizer.optimize()
        team_df = optimizer.format_team_output(best_team)

        # Enriquecer output
        extra = [c for c in
            ['atleta_id','selo_valorizacao','mpv','margem_valorizacao',
             'tipo_confronto','score_h2h','score_cruzado']
            if c in predicoes_df.columns]
        if extra and 'atleta_id' in team_df.columns:
            team_df = team_df.merge(predicoes_df[extra], on='atleta_id', how='left')

        # Imprimir
        imprimir_escalacao_final(team_df, stats, specialist_result, PATRIMONIO)

        # Salvar XLSX
        output_dir = Path(args.output_dir)
        salvar_xlsx(team_df, output_dir, rodada_atual)

    except Exception as e:
        logger.error(f"❌ Erro na otimização: {e}\n{traceback.format_exc()}")
        return

    # ----------------------------------------------------------
    # [EXTRA] RESERVA DE LUXO (opcional)
    # ----------------------------------------------------------
    if args.reserva_luxo:
        logger.info("🔄 [EXTRA] Calculando Reserva de Luxo...")
        try:
            mega_opt = CartolaOptimizer('mega', config={
                'max_players_per_club': opt_config.get('max_mesmo_clube', 3),
                'solver_time_limit': 30,
            })
            from src.features.feature_engineering_v2 import FeatureEngineeringV2
            df_mega = atletas_df.copy()
            if 'fitness_score' in predicoes_df.columns:
                score_map = predicoes_df.set_index('atleta_id')['fitness_score'].to_dict()
                df_mega['mega_score'] = df_mega['atleta_id'].map(score_map).fillna(0)
            else:
                df_mega = FeatureEngineeringV2().engineer_features(df_mega)

            for col in ['selo_valorizacao','mpv','margem_valorizacao']:
                if col in predicoes_df.columns:
                    df_mega = df_mega.merge(
                        predicoes_df[['atleta_id', col]], on='atleta_id', how='left'
                    )

            resultado_reserva = mega_opt.strategy.optimize_with_luxury_reserve(
                df=df_mega, budget=PATRIMONIO, formation=FORMACAO
            )
            if resultado_reserva:
                mega_opt.strategy.print_lineup_with_reserve(resultado_reserva)
                output_dir = Path(args.output_dir)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                reserva_path = output_dir / f"escalacao_reserva_r{rodada_atual}_{ts}.xlsx"
                with pd.ExcelWriter(reserva_path, engine='openpyxl') as w:
                    resultado_reserva['titulares'].to_excel(w, sheet_name='Titulares', index=False)
                    if not resultado_reserva['reservas'].empty:
                        resultado_reserva['reservas'].to_excel(w, sheet_name='Reservas', index=False)
                logger.info(f"💾 Reservas salvas: {reserva_path}")
        except Exception as e:
            logger.error(f"❌ Reserva de Luxo falhou: {e}\n{traceback.format_exc()}")

    print()
    print("=" * 70)
    print("  ✅  PIPELINE CONCLUÍDO! Escalação gerada com análise completa da rodada.")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
