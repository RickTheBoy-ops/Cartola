#!/usr/bin/env python3
"""
Sistema Completo de Análise e Predição do Cartola FC

  Uso simples — roda TUDO e entrega UMA escalação consolidada:
      python main.py

  Opções avançadas (opcionais):
      python main.py --orcamento 110
      python main.py --formacao 4-3-3
      python main.py --modo valorizar
      python main.py --reserva-luxo
      python main.py --v3-features
      python main.py --janela-historico 7

Pipeline obrigatório (sempre executado, sem precisar de flags):
  [1]  Inicializar API
  [2]  Coletar atletas + partidas da rodada
  [3]  Carregar histórico de pontuações + H2H
  [4]  Classificar confrontos Grupo A/B/C via últimos N H2H
  [5]  Desfalques + análise de confrontos
  [6]  Feature Engineering
  [7]  Treinar modelo ML (ou heurística)
  [8]  Predições cruzadas: ML × matchup × H2H × forma recente
  [9]  Specialist Checklist
  [10] Otimizador PuLP  (Programação Linear — solução ÓTIMA)
  [11] Otimizador Genético (heurística complementar)
  [12] ESCALAÇÃO CONSOLIDADA FINAL — melhor dos dois métodos
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
from src.ml.pulp_optimizer import PuLPOptimizer
from src.optimizer.factory import CartolaOptimizer
from src.features.odds_integrator import OddsIntegrator
from src.analysis.matchup_analyzer import MatchupAnalyzer
from src.utils.validators import validar_mercado, validar_formacao, filtrar_atletas_com_jogo

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==============================================================
# CONFIGURAÇÃO
# ==============================================================

def carregar_config(config_path: str = "config.yaml") -> dict:
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"⚠️  {config_path} não encontrado, usando defaults")
        return {}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cartola FC – python main.py gera escalação completa automaticamente"
    )
    p.add_argument("--orcamento",        type=float, default=float(os.getenv("CARTOLA_ORCAMENTO", 110.0)))
    p.add_argument("--formacao",         type=str,   default=os.getenv("CARTOLA_FORMACAO", ""))
    p.add_argument("--modo",             type=str,   default=os.getenv("CARTOLA_MODO", "equilibrado"),
                   choices=["equilibrado", "pontuar", "valorizar"])
    p.add_argument("--output-dir",       type=str,   default="output")
    p.add_argument("--reserva-luxo",     action="store_true", default=False)
    p.add_argument("--v3-features",      action="store_true", default=False)
    p.add_argument("--janela-historico", type=int,   default=5)
    return p.parse_args()


# ==============================================================
# H2H — HISTÓRICO DE CONFRONTOS
# ==============================================================

def carregar_historico_confrontos(conn: sqlite3.Connection, rodada_atual: int) -> pd.DataFrame:
    query = """
        SELECT
            clube_id_a AS clube_a,
            clube_id_b AS clube_b,
            rodada,
            placar_a   AS gols_a,
            placar_b   AS gols_b,
            CASE
                WHEN placar_a > placar_b THEN 'vitoria_a'
                WHEN placar_a < placar_b THEN 'vitoria_b'
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
        logger.warning("⚠️  H2H não disponível ainda (banco novo ou rodadas futuras)")
        return pd.DataFrame(columns=["clube_a","clube_b","rodada","gols_a","gols_b","resultado"])


def ultimos_n_confrontos(hist_df: pd.DataFrame, ca: int, cb: int, n: int) -> pd.DataFrame:
    mask = (
        ((hist_df["clube_a"]==ca) & (hist_df["clube_b"]==cb)) |
        ((hist_df["clube_a"]==cb) & (hist_df["clube_b"]==ca))
    )
    return hist_df[mask].head(n)


def score_h2h(h2h_df: pd.DataFrame, clube_id: int) -> float:
    if h2h_df.empty:
        return 0.5
    pts = 0.0
    for _, row in h2h_df.iterrows():
        if row["clube_a"]==clube_id and row["resultado"]=="vitoria_a":
            pts += 1
        elif row["clube_b"]==clube_id and row["resultado"]=="vitoria_b":
            pts += 1
        elif row["resultado"]=="empate":
            pts += 0.5
    return pts / len(h2h_df)


# ==============================================================
# CLASSIFICAÇÃO DE CONFRONTOS (LEGADO — APENAS LOG)
# ==============================================================

def classificar_confrontos(
    partidas_rodada: pd.DataFrame,
    hist_df: pd.DataFrame,
    janela: int
) -> pd.DataFrame:
    if partidas_rodada.empty:
        return partidas_rodada

    tipos, media_gols, sc_a_list, sc_b_list = [], [], [], []

    for _, row in partidas_rodada.iterrows():
        ca = int(row.get("clube_id_a", row.get("clube_casa_id", 0)))
        cb = int(row.get("clube_id_b", row.get("clube_visitante_id", 0)))
        h2h = ultimos_n_confrontos(hist_df, ca, cb, janela)

        mg = (
            (h2h["gols_a"].fillna(0) + h2h["gols_b"].fillna(0)).mean()
            if not h2h.empty and "gols_a" in h2h.columns else 2.5
        )
        sc_a = score_h2h(h2h, ca)
        sc_b = score_h2h(h2h, cb)
        diff = abs(sc_a - sc_b)

        if diff >= 0.3:   tipo = "A"
        elif mg < 1.5:    tipo = "C"
        else:             tipo = "B"

        tipos.append(tipo)
        media_gols.append(round(mg, 2))
        sc_a_list.append(round(sc_a, 3))
        sc_b_list.append(round(sc_b, 3))

    out = partidas_rodada.copy()
    out["tipo_confronto"] = tipos
    out["media_gols_h2h"] = media_gols
    out["score_h2h_a"]    = sc_a_list
    out["score_h2h_b"]    = sc_b_list
    logger.info(
        f"📊 Confrontos — Grupo A: {tipos.count('A')} | B: {tipos.count('B')} | C: {tipos.count('C')}"
    )
    return out





# ==============================================================
# HELPERS DE SAÍDA
# ==============================================================

def _score_display(row) -> float:
    for col in ["score_cruzado", "score_final", "predicao"]:
        v = row.get(col)
        if v is not None and str(v) != 'nan':
            return float(v)
    return 0.0


def imprimir_escalacao_final(
    team_df: pd.DataFrame,
    stats: dict,
    specialist_result: dict,
    patrimonio: float,
    metodo: str = "CONSOLIDADO"
):
    """Imprime a escalação formatada no terminal."""
    POS_GRUPOS = [
        ([1],       "🧄 GOLEIRO"),
        ([2, 3],    "🛡️  DEFESA"),
        ([4],       "🎯 MEIO-CAMPO"),
        ([5],       "⚽  ATAQUE"),
        ([6],       "👨‍💼 TÉCNICO"),
    ]
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"      🏆  ESCALAÇÃO FINAL [{metodo}]  🏆")
    print(sep)
    pos_col = next((c for c in ["posicao_id", "pos"] if c in team_df.columns), None)
    for ids, label in POS_GRUPOS:
        if pos_col:
            subset = team_df[team_df[pos_col].isin(ids)]
        else:
            subset = team_df
        if subset.empty:
            continue
        print(f"\n{label}")
        for _, r in subset.iterrows():
            nome    = r.get("apelido", r.get("Atleta", r.get("nome", "?")))
            preco   = float(r.get("preco", r.get("Preço (C$)", 0)))
            score   = _score_display(r)
            tipo_c  = r.get("tipo_confronto", "-")
            h2h_val = float(r.get("score_h2h", 0))
            mult    = float(r.get("mult_confronto", 1.0))
            atk_mul = float(r.get("attack_multiplier", 1.0))
            def_mul = float(r.get("defense_multiplier", 1.0))
            win_p   = float(r.get("win_probability", 0.0))
            cs_p    = float(r.get("clean_sheet_probability", 0.0))
            print(
                f"   {nome:<24}  C${preco:>6.2f}  "
                f"Score:{score:>7.2f}  "
                f"Conf:{tipo_c}(x{mult:.2f})  "
                f"H2H:{h2h_val:.2f}  "
                f"AtkMul:{atk_mul:.2f} DefMul:{def_mul:.2f} "
                f"Win:{win_p:.0%} SG:{cs_p:.0%}"
            )
    print(f"\n{'━' * 72}")
    print(f"  ⚽ Pontos Preditos  : {stats.get('total_pontos_preditos', 0):.2f}")
    print(f"  💰 Preço Total      : C${stats.get('total_preco', 0):.2f} / C${patrimonio:.2f} "
          f"({stats.get('patrimonio_usado_pct', 0):.1f}%)")
    cap = specialist_result.get('recommended_captain', '')
    if cap:
        print(f"  ⭐ Capitão Sugerido : {cap}")
    print(sep + "\n")


def salvar_xlsx(team_df: pd.DataFrame, output_dir: Path, rodada: int, sufixo: str = "") -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    nome = f"escalacao_r{rodada}_{sufixo}_{ts}.xlsx" if sufixo else f"escalacao_r{rodada}_{ts}.xlsx"
    path = output_dir / nome
    team_df.to_excel(path, index=False)
    logger.info(f"💾 Escalação salva: {path}")
    return path


# ==============================================================
# PIPELINE PRINCIPAL
# ==============================================================

def main():
    args   = parse_args()
    config = carregar_config()

    print()
    print("=" * 72)
    print("  🏆  CARTOLA FC | Pipeline Completo | python main.py")
    print(f"  Orçamento: C${args.orcamento:.0f} | Modo: {args.modo.upper()} | H2H: {args.janela_historico} rodadas")
    print("=" * 72)

    EMAIL      = os.getenv('CARTOLA_EMAIL')
    PASSWORD   = os.getenv('CARTOLA_PASSWORD')
    PATRIMONIO = args.orcamento
    FORMACAO   = args.formacao if args.formacao else '4-3-3'
    JANELA     = args.janela_historico

    if not validar_formacao(FORMACAO):
        logger.error(f"❌ Formação '{FORMACAO}' inválida.")
        return

    # ----------------------------------------------------------
    # [1/12] API
    # ----------------------------------------------------------
    logger.info("📡 [1/12] Inicializando cliente API...")
    api_client = CartolaAPIClient(email=EMAIL, password=PASSWORD)

    # ----------------------------------------------------------
    # [2/12] COLETA
    # ----------------------------------------------------------
    logger.info("📥 [2/12] Coletando dados da rodada...")
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
    # [3/12] HISTÓRICO + H2H
    # ----------------------------------------------------------
    logger.info("📚 [3/12] Carregando histórico + H2H...")
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
        f"📊 Histórico: {len(historico_df)} reg | Partidas: {len(partidas_df)} | H2H: {len(hist_h2h_df)}"
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
    # [4/12] CLASSIFICAÇÃO DE CONFRONTOS (LOG LEGADO)
    # ----------------------------------------------------------
    logger.info("🔍 [4/12] Classificando confrontos (Grupo A/B/C) para log...")
    partidas_rodada = (
        partidas_df[partidas_df['rodada'] == rodada_atual].copy()
        if not partidas_df.empty else pd.DataFrame()
    )
    partidas_enriquecidas = classificar_confrontos(partidas_rodada, hist_h2h_df, JANELA)
    if not partidas_enriquecidas.empty:
        cols_show = [c for c in
            ["clube_id_a","clube_id_b","tipo_confronto","media_gols_h2h","score_h2h_a","score_h2h_b"]
            if c in partidas_enriquecidas.columns]
        logger.info(f"\n{partidas_enriquecidas[cols_show].to_string(index=False)}")

    # ----------------------------------------------------------
    # [5/12] FEATURE ENGINEERING
    # ----------------------------------------------------------
    logger.info("🔧 [5/12] Gerando features...")
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
    # [6/12] ODDS
    # ----------------------------------------------------------
    logger.info("🎲 [6/12] Integrando odds...")
    try:
        odds_integrator = OddsIntegrator()
        atletas_df = odds_integrator.enrich(atletas_df, partidas_df, clubes_df)
        logger.info("✅ Odds integradas")
    except Exception as e:
        logger.warning(f"⚠️  Odds indisponíveis ({e}) — continuando sem odds")

    # ----------------------------------------------------------
    # [7/12] TREINO ML
    # ----------------------------------------------------------
    logger.info("🧠 [7/12] Treinando modelo ML...")
    predictor    = CartolaPredictor(model_type=config.get('ml',{}).get('model_type','rf'))
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
        logger.warning(f"⚠️  Histórico insuficiente ({len(dados_treino)} reg) — usando heurística")

    # ----------------------------------------------------------
    # [8/12] PREDIÇÕES + MATCHUP ANALYZER
    # ----------------------------------------------------------
    logger.info(f"🎯 [8/12] Predições cruzadas (ML + MatchupAnalyzer)...")

    if can_train and predictor.is_trained:
        ultimos = (
            historico_df.sort_values('rodada')
            .groupby('atleta_id').tail(JANELA)
        )
        ultimos_agg = ultimos.groupby('atleta_id').last().reset_index()
        dados_pred = atletas_df.merge(
            ultimos_agg, on='atleta_id', how='left', suffixes=('','_hist')
        ).fillna(0)
        predicoes_df = predictor.predict_full(dados_pred)
        logger.info("✅ Predições ML geradas")
    else:
        predicoes_df = CartolaPredictor.fallback_heuristica(atletas_df)
        predicoes_df['predicao_ajustada'] = predicoes_df['predicao']
        predicoes_df = CartolaPredictor.add_valorizacao_flag(predicoes_df)

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

    logger.info(f"⚖️  Modo {args.modo.upper()} aplicado")

    # Garante que predicoes_df tenha clube_id e posicao_id
    if 'clube_id' not in predicoes_df.columns \
            and 'atleta_id' in predicoes_df.columns \
            and 'clube_id' in atletas_df.columns:
        mapa_clube = (
            atletas_df
            .set_index('atleta_id')['clube_id']
            .to_dict()
        )
        predicoes_df['clube_id'] = (
            predicoes_df['atleta_id']
            .map(mapa_clube)
            .fillna(0)
            .astype(int)
        )
    if 'posicao_id' not in predicoes_df.columns \
            and 'atleta_id' in predicoes_df.columns \
            and 'posicao_id' in atletas_df.columns:
        mapa_pos = (
            atletas_df
            .set_index('atleta_id')['posicao_id']
            .to_dict()
        )
        predicoes_df['posicao_id'] = (
            predicoes_df['atleta_id']
            .map(mapa_pos)
            .fillna(0)
            .astype(int)
        )

    # MatchupAnalyzer: multipliers por clube
    matchups_df = pd.DataFrame()
    try:
        odds_por_clube = None
        if 'prob_gol' in atletas_df.columns and 'prob_sg' in atletas_df.columns:
            odds_por_clube = (
                atletas_df
                .groupby('clube_id')[['prob_gol', 'prob_sg']]
                .mean()
                .reset_index()
            )

        matchup = MatchupAnalyzer(
            rodada=rodada_atual,
            partidas_df=partidas_df,
            historico_partidas_df=partidas_df,
            odds_por_clube=odds_por_clube,
            janela_recente=JANELA,
        )
        matchups_df = matchup.build_team_matchups()
        if not matchups_df.empty:
            predicoes_df = MatchupAnalyzer.apply_to_predictions(predicoes_df, matchups_df)
            ranking_col = 'score_cruzado'
            logger.info("✅ MatchupAnalyzer aplicado com sucesso")
        else:
            ranking_col = 'score_final'
            predicoes_df['score_cruzado'] = predicoes_df['score_final']
            logger.warning("⚠️ MatchupAnalyzer retornou vazio, usando score_final como score_cruzado")
    except Exception as e:
        logger.error(f"❌ Erro no MatchupAnalyzer: {e}\\n{traceback.format_exc()}")
        ranking_col = 'score_final'
        predicoes_df['score_cruzado'] = predicoes_df['score_final']

    predicoes_df['fitness_score'] = predicoes_df[ranking_col]

    top10 = predicoes_df.nlargest(10, ranking_col)
    cols_top = [c for c in [
        'apelido','posicao_id','preco',ranking_col,
        'attack_multiplier','defense_multiplier','win_probability','clean_sheet_probability'
    ] if c in top10.columns]
    logger.info(f"\n🏆 Top 10 pré-otimização:\n{top10[cols_top].to_string(index=False)}")

    # ----------------------------------------------------------
    # [9/12] SPECIALIST CHECKLIST
    # ----------------------------------------------------------
    logger.info("🔎 [9/12] Specialist Checklist...")
    specialist_result: dict = {}
    try:
        from src.ml.specialist_logic import CartolaPrescalingChecklist
        checklist = CartolaPrescalingChecklist(
            rodada=rodada_atual,
            budget=PATRIMONIO,
            atletas_df=atletas_df,
            partidas_df=(
                partidas_enriquecidas
                if not partidas_enriquecidas.empty else partidas_df
            ),
            predicoes_df=predicoes_df,
            modo=args.modo,
            matchups_df=matchups_df,
        )
        specialist_result = checklist.run()
        logger.info(
            f"✅ Checklist OK | Capitão: {specialist_result.get('recommended_captain','N/A')}"
        )
        for w in specialist_result.get('warnings', []):
            logger.warning(f"   ⚠️  {w}")
    except Exception as e:
        logger.error(f"❌ Specialist falhou: {e}\n{traceback.format_exc()}")
        # Não interrompe o pipeline

    # Merge atletas + predicoes para os otimizadores
    pred_opt = predicoes_df.copy()
    if 'predicao_std' not in pred_opt.columns:
        pred_opt['predicao_std'] = 0.0
    if 'odds_score' in pred_opt.columns:
        pred_opt['fitness_score'] = (
            pred_opt['fitness_score'] * 0.8 + pred_opt['odds_score'] * 20 * 0.2
        )

    opt_config = config.get('optimizer', {})

    # ----------------------------------------------------------
    # [10/12] OTIMIZADOR PuLP — PROGRAMAÇÃO LINEAR (solutão ÓTIMA)
    # ----------------------------------------------------------
    logger.info("📊 [10/12] Otimizador PuLP (Programação Linear)...")
    pulp_team_df, pulp_stats = None, {}

    if PuLPOptimizer.disponivel():
        try:
            merge_cols = [
                'atleta_id',
                'score_cruzado',
                'score_final',
                'attack_multiplier',
                'defense_multiplier',
                'win_probability',
                'clean_sheet_probability',
            ]
            merge_cols = [c for c in merge_cols if c in pred_opt.columns]
            atletas_enriquecidos = atletas_df.merge(
                pred_opt[merge_cols].drop_duplicates('atleta_id'),
                on='atleta_id',
                how='left',
            )
            pulp_opt = PuLPOptimizer(
                atletas_df=atletas_enriquecidos,
                patrimonio=PATRIMONIO,
                formacao=FORMACAO,
                max_mesmo_clube=opt_config.get('max_mesmo_clube', 3),
                score_col='score_cruzado',
            )
            pulp_team_df, pulp_stats = pulp_opt.optimize()
            if pulp_team_df is not None:
                logger.info(
                    f"✅ PuLP OK | Score: {pulp_stats.get('total_score_cruzado',0):.2f} | "
                    f"Preço: C${pulp_stats.get('total_preco',0):.2f}"
                )
        except Exception as e:
            logger.error(f"❌ PuLP falhou: {e}\n{traceback.format_exc()}")
    else:
        logger.warning("⚠️  PuLP não instalado. Instale com: pip install pulp")

    # ----------------------------------------------------------
    # [11/12] OTIMIZADOR GENÉTICO (heurística complementar)
    # ----------------------------------------------------------
    logger.info("🧬 [11/12] Otimizador Genético...")
    genetic_team, genetic_stats = None, {}
    try:
        gen_opt = GeneticTeamOptimizer(
            atletas_df=atletas_df,
            predicoes=pred_opt,
            patrimonio=PATRIMONIO,
            formacao=FORMACAO,
            population_size=opt_config.get('population_size', 250),
            generations=opt_config.get('generations', 150),
            mutation_rate=opt_config.get('mutation_rate', 0.20),
            elite_size=opt_config.get('elite_size', 20),
            max_mesmo_clube=opt_config.get('max_mesmo_clube', 3),
            penalidade_variancia=opt_config.get('penalidade_variancia', True),
        )
        best_team_raw, genetic_stats = gen_opt.optimize()
        genetic_team = gen_opt.format_team_output(best_team_raw)

        # Enriquecer com scores cruzados
        extra = [c for c in [
            'atleta_id',
            'score_cruzado',
            'selo_valorizacao',
            'attack_multiplier','defense_multiplier',
            'win_probability','clean_sheet_probability',
        ] if c in predicoes_df.columns]
        if extra and 'atleta_id' in genetic_team.columns:
            genetic_team = genetic_team.merge(
                predicoes_df[extra], on='atleta_id', how='left'
            )
        logger.info(
            f"✅ Genético OK | Pontos: {genetic_stats.get('total_pontos_preditos',0):.2f} | "
            f"Preço: C${genetic_stats.get('total_preco',0):.2f}"
        )
    except Exception as e:
        logger.error(f"❌ Genético falhou: {e}\n{traceback.format_exc()}")

    # ----------------------------------------------------------
    # [12/12] ESCALAÇÃO CONSOLIDADA — Melhor dos dois
    # ----------------------------------------------------------
    logger.info("🏆 [12/12] Consolidando escalação final...")
    output_dir = Path(args.output_dir)

    pulp_score    = pulp_stats.get('total_score_cruzado', -1) if pulp_team_df is not None else -1
    genetic_score = genetic_stats.get('total_pontos_preditos', -1) if genetic_team is not None else -1

    # Mostra ambos se disponíveis, elege o melhor
    if pulp_team_df is not None and not pulp_team_df.empty:
        ps = dict(pulp_stats)
        ps['patrimonio_usado_pct'] = ps.get('patrimonio_usado_pct', 0)
        imprimir_escalacao_final(pulp_team_df, ps, specialist_result, PATRIMONIO, metodo="PuLP (ÓTIMO)")
        salvar_xlsx(pulp_team_df, output_dir, rodada_atual, "pulp")

    if genetic_team is not None and not genetic_team.empty:
        gs = dict(genetic_stats)
        # Calcula % do patrimônio usado a partir do preço total
        total_preco_gen = gs.get('total_preco', 0)
        gs['patrimonio_usado_pct'] = (
            (total_preco_gen / PATRIMONIO) * 100
            if PATRIMONIO else 0
        )
        imprimir_escalacao_final(genetic_team, gs, specialist_result, PATRIMONIO, metodo="Genético")
        salvar_xlsx(genetic_team, output_dir, rodada_atual, "genetico")

    # ESCALAÇÃO FINAL: PuLP tem prioridade (matematicamente ótimo)
    # Se PuLP não rodou, usa genético.
    if pulp_team_df is not None and not pulp_team_df.empty:
        final_team  = pulp_team_df
        final_stats = pulp_stats
        metodo_final = "PuLP — Matematicamente Ótimo"
    elif genetic_team is not None:
        final_team  = genetic_team
        final_stats = genetic_stats
        metodo_final = "Genético (PuLP indisponível)"
    else:
        logger.error("❌ Nenhum otimizador produziu resultado.")
        return

    print()
    print("#" * 72)
    print(f"#  ESCALAÇÃO CONSOLIDADA FINAL   [{metodo_final}]")
    print("#" * 72)
    imprimir_escalacao_final(final_team, final_stats, specialist_result, PATRIMONIO, metodo="FINAL")
    salvar_xlsx(final_team, output_dir, rodada_atual, "FINAL")

    # ----------------------------------------------------------
    # [EXTRA] RESERVA DE LUXO
    # ----------------------------------------------------------
    if args.reserva_luxo:
        logger.info("🔄 [EXTRA] Calculando Reserva de Luxo...")
        try:
            mega_opt = CartolaOptimizer('mega', config={
                'max_players_per_club': opt_config.get('max_mesmo_clube', 3),
                'solver_time_limit': 30,
            })
            df_mega = atletas_df.copy()
            if 'fitness_score' in predicoes_df.columns:
                score_map = predicoes_df.set_index('atleta_id')['fitness_score'].to_dict()
                df_mega['mega_score'] = df_mega['atleta_id'].map(score_map).fillna(0)
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
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                with pd.ExcelWriter(
                    output_dir / f"escalacao_reserva_r{rodada_atual}_{ts}.xlsx",
                    engine='openpyxl'
                ) as w:
                    resultado_reserva['titulares'].to_excel(w, sheet_name='Titulares', index=False)
                    if not resultado_reserva['reservas'].empty:
                        resultado_reserva['reservas'].to_excel(w, sheet_name='Reservas', index=False)
                logger.info("✅ Reservas salvas")
        except Exception as e:
            logger.error(f"❌ Reserva de Luxo falhou: {e}")

    print()
    print("=" * 72)
    print("  ✅  PIPELINE CONCLUÍDO! Escalação gerada com análise completa.")
    print(f"  📁 Salva em: {output_dir}/")
    print("=" * 72)
    print()


if __name__ == "__main__":
    main()
