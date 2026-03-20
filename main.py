#!/usr/bin/env python3
"""
Sistema Completo de Análise e Predição do Cartola FC

Pipeline OBRIGATÓRIO (em ordem):
  1. Inicializar API e coletar rodada atual
  2. Coletar atletas + partidas
  3. Carregar histórico dos últimos 5 confrontos entre os times
  4. Analisar confrontos (Grupo A/B/C) + desfalques
  5. Feature Engineering (V2 ou V3)
  6. Integrar odds (V3 only)
  7. Treinar modelo ML (ou fallback heurística)
  8. Gerar predições com cruzamento: ML + matchup + histórico + forma
  9. Specialist Checklist (validação obrigatória)
  10. Otimização genética → escalação final
  11. Reserva de Luxo (opcional)

Uso:
    python main.py                                 # usa .env / defaults
    python main.py --orcamento 110 --formacao 4-3-3
    python main.py --modo valorizar --reserva-luxo --v3-features
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

# Imports do projeto
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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURAÇÃO
# ============================================================

def carregar_config(config_path: str = "config.yaml") -> dict:
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"⚠️  {config_path} não encontrado, usando defaults")
        return {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cartola FC Optimizer - Pipeline completo de predição e escalação"
    )
    parser.add_argument("--orcamento", type=float,
                        default=float(os.getenv("CARTOLA_ORCAMENTO", 110.0)))
    parser.add_argument("--formacao", type=str,
                        default=os.getenv("CARTOLA_FORMACAO", ""))
    parser.add_argument("--modo", type=str,
                        default=os.getenv("CARTOLA_MODO", "equilibrado"),
                        choices=["equilibrado", "pontuar", "valorizar"])
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--reserva-luxo", action="store_true", default=False)
    parser.add_argument("--v3-features", action="store_true", default=False)
    parser.add_argument("--janela-historico", type=int, default=5,
                        help="Número de rodadas/confrontos no histórico (padrão: 5)")
    return parser.parse_args()


# ============================================================
# ETAPA 3b – HISTÓRICO DE CONFRONTOS (últimos N)
# ============================================================

def carregar_historico_confrontos(
    conn: sqlite3.Connection,
    rodada_atual: int,
    janela: int = 5
) -> pd.DataFrame:
    """
    Para cada par de times que se enfrentam na rodada atual,
    busca os últimos `janela` resultados diretos no banco.
    Retorna DataFrame com colunas:
      clube_a, clube_b, rodada, gols_a, gols_b, resultado
    """
    query = """
        SELECT
            p.clube_id_a   AS clube_a,
            p.clube_id_b   AS clube_b,
            p.rodada,
            p.placar_a     AS gols_a,
            p.placar_b     AS gols_b,
            CASE
                WHEN p.placar_a > p.placar_b THEN 'vitoria_a'
                WHEN p.placar_a < p.placar_b THEN 'vitoria_b'
                ELSE 'empate'
            END AS resultado
        FROM partidas p
        WHERE p.rodada < ?
        ORDER BY p.rodada DESC
    """
    try:
        df = pd.read_sql_query(query, conn, params=(rodada_atual,))
    except Exception:
        logger.warning("⚠️  Tabela 'partidas' sem colunas placar_a/placar_b – histórico de confrontos parcial")
        df = pd.DataFrame(columns=["clube_a", "clube_b", "rodada", "gols_a", "gols_b", "resultado"])
    return df


def ultimos_n_confrontos(
    historico_df: pd.DataFrame,
    clube_a: int,
    clube_b: int,
    n: int = 5
) -> pd.DataFrame:
    """
    Filtra os últimos N jogos diretos entre clube_a e clube_b
    (considerando os dois sentidos do confronto).
    """
    mask = (
        ((historico_df["clube_a"] == clube_a) & (historico_df["clube_b"] == clube_b)) |
        ((historico_df["clube_a"] == clube_b) & (historico_df["clube_b"] == clube_a))
    )
    return historico_df[mask].head(n)


def calcular_score_historico_confronto(
    historico_confronto: pd.DataFrame,
    clube_id: int
) -> float:
    """
    Gera um score 0-1 baseado no aproveitamento do clube_id
    nos últimos confrontos diretos.
    - 1.0 = ganhou todos
    - 0.5 = equilibrado
    - 0.0 = perdeu todos
    """
    if historico_confronto.empty:
        return 0.5

    vitorias = 0
    total = len(historico_confronto)

    for _, row in historico_confronto.iterrows():
        if row["clube_a"] == clube_id and row["resultado"] == "vitoria_a":
            vitorias += 1
        elif row["clube_b"] == clube_id and row["resultado"] == "vitoria_b":
            vitorias += 1
        elif row["resultado"] == "empate":
            vitorias += 0.5

    return vitorias / total if total > 0 else 0.5


# ============================================================
# ETAPA 4 – ANÁLISE DE CONFRONTOS (Grupo A/B/C)
# ============================================================

def classificar_confrontos(
    partidas_df: pd.DataFrame,
    historico_confrontos_df: pd.DataFrame,
    janela: int = 5
) -> pd.DataFrame:
    """
    Enriquece partidas_df com:
      - tipo_confronto: 'A' (favoritismo), 'B' (aberto), 'C' (truncado)
      - media_gols_historico: média de gols nos últimos N H2H
      - score_historico_a / score_historico_b
    """
    if partidas_df.empty:
        return partidas_df

    tipos, media_gols, sc_a_list, sc_b_list = [], [], [], []

    for _, row in partidas_df.iterrows():
        ca = int(row.get("clube_id_a", row.get("clube_casa", 0)))
        cb = int(row.get("clube_id_b", row.get("clube_visitante", 0)))

        h2h = ultimos_n_confrontos(historico_confrontos_df, ca, cb, janela)

        # Média de gols no H2H
        if not h2h.empty and "gols_a" in h2h.columns:
            mg = (h2h["gols_a"].fillna(0) + h2h["gols_b"].fillna(0)).mean()
        else:
            mg = 2.5  # default neutro

        sc_a = calcular_score_historico_confronto(h2h, ca)
        sc_b = calcular_score_historico_confronto(h2h, cb)

        # Classificação
        diff = abs(sc_a - sc_b)
        if diff >= 0.3:
            tipo = "A"  # Amplo favoritismo
        elif mg < 1.5:
            tipo = "C"  # Jogo truncado
        else:
            tipo = "B"  # Aberto / equilibrado

        tipos.append(tipo)
        media_gols.append(round(mg, 2))
        sc_a_list.append(round(sc_a, 3))
        sc_b_list.append(round(sc_b, 3))

    partidas_df = partidas_df.copy()
    partidas_df["tipo_confronto"] = tipos
    partidas_df["media_gols_h2h"] = media_gols
    partidas_df["score_h2h_a"] = sc_a_list
    partidas_df["score_h2h_b"] = sc_b_list

    logger.info(
        f"📊 Confrontos classificados – "
        f"Grupo A: {tipos.count('A')} | B: {tipos.count('B')} | C: {tipos.count('C')}"
    )
    return partidas_df


# ============================================================
# ETAPA 5 – CRUZAMENTO DE SCORES
# ============================================================

def enriquecer_predicoes_com_confronto(
    predicoes_df: pd.DataFrame,
    atletas_df: pd.DataFrame,
    partidas_enriquecidas: pd.DataFrame,
    historico_confrontos_df: pd.DataFrame,
    janela: int = 5
) -> pd.DataFrame:
    """
    Cruza as predições ML com:
      1. Tipo do confronto (A/B/C)
      2. Score H2H do time do atleta
      3. Forma recente do atleta (média últimas N rodadas)
      4. Penalidade para atletas em confrontos Grupo C

    Adiciona coluna `score_cruzado` que substitui `score_final`
    no ranqueamento para otimização.
    """
    if predicoes_df.empty:
        return predicoes_df

    # Mapear clube → tipo confronto e score H2H
    clube_tipo: Dict[int, str] = {}
    clube_score_h2h: Dict[int, float] = {}

    if not partidas_enriquecidas.empty:
        for _, row in partidas_enriquecidas.iterrows():
            ca = int(row.get("clube_id_a", row.get("clube_casa", 0)))
            cb = int(row.get("clube_id_b", row.get("clube_visitante", 0)))
            tipo = row.get("tipo_confronto", "B")
            clube_tipo[ca] = tipo
            clube_tipo[cb] = tipo
            clube_score_h2h[ca] = row.get("score_h2h_a", 0.5)
            clube_score_h2h[cb] = row.get("score_h2h_b", 0.5)

    pred = predicoes_df.copy()
    score_base = pred.get("score_final", pred.get("predicao_ajustada", pred.get("predicao", pd.Series(0, index=pred.index))))

    # Garantir coluna clube_id no df de predições
    if "clube_id" not in pred.columns and "atleta_id" in pred.columns and "clube_id" in atletas_df.columns:
        clube_map = atletas_df.set_index("atleta_id")["clube_id"].to_dict()
        pred["clube_id"] = pred["atleta_id"].map(clube_map).fillna(0).astype(int)

    tipo_col, h2h_col = [], []
    for _, row in pred.iterrows():
        cid = int(row.get("clube_id", 0))
        tipo_col.append(clube_tipo.get(cid, "B"))
        h2h_col.append(clube_score_h2h.get(cid, 0.5))

    pred["tipo_confronto"] = tipo_col
    pred["score_h2h"] = h2h_col

    # Multiplicadores por tipo de confronto
    mult_map = {"A": 1.15, "B": 1.0, "C": 0.85}
    pred["mult_confronto"] = pred["tipo_confronto"].map(mult_map).fillna(1.0)

    # Score cruzado = ML × confronto × H2H (normalizado 0.7-1.3)
    h2h_norm = 0.7 + pred["score_h2h"] * 0.6  # escala 0.7 a 1.3
    pred["score_cruzado"] = score_base * pred["mult_confronto"] * h2h_norm

    logger.info(
        f"🔀 Score cruzado gerado para {len(pred)} atletas "
        f"(confronto + H2H últimos {janela})"
    )
    return pred


# ============================================================
# HELPERS DE SAÍDA
# ============================================================

def salvar_escalacao_xlsx(team_df: pd.DataFrame, output_dir: Path, rodada: int, sufixo: str = "") -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    nome = f"escalacao_rodada{rodada}_{sufixo}_{ts}.xlsx" if sufixo else f"escalacao_rodada{rodada}_{ts}.xlsx"
    caminho = output_dir / nome
    team_df.to_excel(caminho, index=False)
    logger.info(f"💾 Escalação salva em: {caminho}")
    return caminho


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

def main():
    args = parse_args()
    config = carregar_config()

    logger.info("=" * 70)
    logger.info("🏆 SISTEMA DE ANÁLISE CARTOLA FC – PIPELINE COMPLETO")
    logger.info(f"⚙️  Orçamento: C${args.orcamento:.1f} | Modo: {args.modo}")
    logger.info(f"⚙️  Features V3: {args.v3_features} | Reserva de Luxo: {args.reserva_luxo}")
    logger.info(f"⚙️  Janela histórico confrontos: {args.janela_historico} rodadas")
    logger.info("=" * 70)

    EMAIL = os.getenv('CARTOLA_EMAIL')
    PASSWORD = os.getenv('CARTOLA_PASSWORD')
    PATRIMONIO = args.orcamento
    FORMACAO = args.formacao if args.formacao else None
    JANELA = args.janela_historico

    if FORMACAO and not validar_formacao(FORMACAO):
        logger.error(f"❌ Formação '{FORMACAO}' inválida.")
        return

    # ----------------------------------------------------------
    # ETAPA 1 – CLIENTE API
    # ----------------------------------------------------------
    logger.info("\n📡 [1/10] Inicializando cliente API...")
    api_client = CartolaAPIClient(email=EMAIL, password=PASSWORD)

    # ----------------------------------------------------------
    # ETAPA 2 – COLETAR DADOS DA RODADA
    # ----------------------------------------------------------
    logger.info("\n📥 [2/10] Coletando dados da API...")
    collector = CartolaDataCollector(api_client)

    try:
        mercado = collector.collect_mercado_status()
        mercado_info = validar_mercado(mercado)
        if not mercado_info['valido']:
            logger.error(f"❌ {mercado_info['mensagem']}")
            return

        rodada_atual = mercado_info['rodada_atual']
        logger.info(f"📊 {mercado_info['mensagem']}")

        atletas_df = collector.collect_atletas_mercado(rodada_atual)
        logger.info(f"✅ Coletados {len(atletas_df)} atletas")

        if len(atletas_df) == 0:
            logger.error("❌ Nenhum atleta encontrado.")
            return

        collector.collect_partidas(rodada_atual)

    except Exception as e:
        logger.error(f"❌ Erro na coleta de dados: {e}")
        return

    # ----------------------------------------------------------
    # ETAPA 3 – DADOS HISTÓRICOS + CONFRONTOS
    # ----------------------------------------------------------
    logger.info("\n📚 [3/10] Preparando dados históricos...")

    conn = sqlite3.connect(collector.db_path)

    historico_df = pd.read_sql_query("""
        SELECT p.*, a.clube_id FROM pontuacoes p
        JOIN atletas a ON p.atleta_id = a.atleta_id
        WHERE p.rodada >= ? AND p.rodada <= ?
        ORDER BY p.atleta_id, p.rodada
    """, conn, params=(max(1, rodada_atual - 15), rodada_atual))

    partidas_df = pd.read_sql_query("""
        SELECT * FROM partidas
        WHERE rodada >= ? AND rodada <= ?
    """, conn, params=(max(1, rodada_atual - 15), rodada_atual))

    # Histórico completo para cruzamento H2H
    historico_confrontos_df = carregar_historico_confrontos(conn, rodada_atual, JANELA)

    conn.close()

    logger.info(
        f"📊 Histórico: {len(historico_df)} registros | "
        f"Partidas: {len(partidas_df)} | "
        f"H2H base: {len(historico_confrontos_df)} jogos"
    )

    # Clubes
    try:
        r = requests.get("https://api.cartola.globo.com/clubes",
                         headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        clubes_raw = r.json()
        clubes_df = pd.DataFrame([
            {'id': int(k), 'nome': v.get('nome', ''), 'abreviacao': v.get('abreviacao', '')}
            for k, v in clubes_raw.items()
        ])
        logger.info(f"✅ {len(clubes_df)} clubes carregados da API")
    except Exception:
        clubes_df = atletas_df[['clube_id']].drop_duplicates().rename(columns={'clube_id': 'id'})
        clubes_df['nome'] = clubes_df['id'].astype(str)
        logger.warning("⚠️  Clubes via fallback (apenas IDs)")

    # Filtrar atletas com jogo confirmado
    atletas_df = filtrar_atletas_com_jogo(atletas_df, rodada_atual, partidas_df)
    logger.info(f"✅ {len(atletas_df)} atletas com jogo confirmado na rodada {rodada_atual}")

    # ----------------------------------------------------------
    # ETAPA 4 – ANÁLISE E CLASSIFICAÇÃO DE CONFRONTOS (A/B/C)
    # ----------------------------------------------------------
    logger.info("\n🔍 [4/10] Analisando confrontos da rodada (Grupo A/B/C)...")

    # Filtrar apenas partidas da rodada atual
    partidas_rodada = partidas_df[partidas_df['rodada'] == rodada_atual].copy() \
        if not partidas_df.empty else pd.DataFrame()

    partidas_enriquecidas = classificar_confrontos(
        partidas_rodada, historico_confrontos_df, JANELA
    )

    if not partidas_enriquecidas.empty:
        logger.info("\n" + partidas_enriquecidas[
            [c for c in ["clube_id_a", "clube_id_b", "tipo_confronto", "media_gols_h2h"]
             if c in partidas_enriquecidas.columns]
        ].to_string(index=False))

    # ----------------------------------------------------------
    # ETAPA 5 – FEATURE ENGINEERING
    # ----------------------------------------------------------
    logger.info("\n🔧 [5/10] Criando features...")

    if len(historico_df) > 0:
        if args.v3_features:
            historico_df = FeatureEngineer.engineer_all_features_v3(historico_df, partidas_df)
            logger.info("✅ Feature Engineering V3 aplicado")
        else:
            historico_df = FeatureEngineer.engineer_all_features(historico_df, partidas_df)
            logger.info("✅ Feature Engineering V2 aplicado")
    else:
        logger.warning("⚠️  Sem histórico para feature engineering.")

    # ----------------------------------------------------------
    # ETAPA 6 – INTEGRAR ODDS (V3 only)
    # ----------------------------------------------------------
    if args.v3_features:
        logger.info("🎲 [6/10] Integrando odds de probabilidade...")
        try:
            odds_integrator = OddsIntegrator()
            atletas_df = odds_integrator.enrich(atletas_df, partidas_df, clubes_df)
            logger.info("✅ Odds integradas")
        except Exception as e:
            logger.warning(f"⚠️  Odds indisponíveis: {e}")
    else:
        logger.info("⏭️  [6/10] Odds puladas (use --v3-features para ativar)")

    # ----------------------------------------------------------
    # ETAPA 7 – TREINAR MODELO ML
    # ----------------------------------------------------------
    logger.info("\n🧠 [7/10] Treinando modelo de predição...")

    predictor = CartolaPredictor(
        model_type=config.get('ml', {}).get('model_type', 'rf')
    )
    dados_treino = historico_df[historico_df['rodada'] < rodada_atual] \
        if len(historico_df) > 0 else pd.DataFrame()
    can_train = len(dados_treino) >= CartolaPredictor.HISTORICO_MINIMO

    if can_train:
        metrics = predictor.train(dados_treino, validate=True)
        model_path = Path("data/models/cartola_predictor.pkl")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        predictor.save_model(str(model_path))
        logger.info(f"✅ Modelo treinado | métricas: {metrics}")
    else:
        logger.warning(
            f"⚠️  Histórico insuficiente ({len(dados_treino)} registros). Usando heurística."
        )

    # ----------------------------------------------------------
    # ETAPA 8 – PREDIÇÕES + CRUZAMENTO HISTÓRICO
    # ----------------------------------------------------------
    logger.info(f"\n🎯 [8/10] Predizendo + cruzando histórico (últimos {JANELA} confrontos)...")

    if can_train and predictor.is_trained:
        ultimos_dados = historico_df.sort_values('rodada').groupby('atleta_id').tail(JANELA)
        ultimos_agregados = (
            ultimos_dados.sort_values('rodada')
            .groupby('atleta_id').last()
            .reset_index()
        )
        dados_predicao = atletas_df.merge(
            ultimos_agregados, on='atleta_id', how='left', suffixes=('', '_hist')
        ).fillna(0)
        predicoes_df = predictor.predict_full(dados_predicao)
        logger.info("✅ Predições com pesos táticos geradas")
    else:
        predicoes_df = CartolaPredictor.fallback_heuristica(atletas_df)
        predicoes_df['predicao_ajustada'] = predicoes_df['predicao']
        predicoes_df = CartolaPredictor.add_valorizacao_flag(predicoes_df)

    # Score por modo
    score_col = 'predicao_ajustada' if 'predicao_ajustada' in predicoes_df.columns else 'predicao'
    if args.modo == 'valorizar' and 'valorizacao_score' in predicoes_df.columns:
        predicoes_df['score_final'] = (
            predicoes_df[score_col] * 0.4 + predicoes_df['valorizacao_score'] * 0.6
        )
        logger.info("🏷️  Modo VALORIZAR")
    elif args.modo == 'pontuar':
        predicoes_df['score_final'] = predicoes_df[score_col]
        logger.info("⚽ Modo PONTUAR")
    else:
        val_weight = predicoes_df.get(
            'valorizacao_score', pd.Series(50, index=predicoes_df.index)
        )
        predicoes_df['score_final'] = (
            predicoes_df[score_col] * 0.65 + val_weight * 0.35
        )
        logger.info("⚖️  Modo EQUILIBRADO")

    # === CRUZAMENTO HISTÓRICO DE CONFRONTOS ===
    predicoes_df = enriquecer_predicoes_com_confronto(
        predicoes_df, atletas_df, partidas_enriquecidas,
        historico_confrontos_df, JANELA
    )

    # Usar score_cruzado se disponível, senão score_final
    ranking_col = 'score_cruzado' if 'score_cruzado' in predicoes_df.columns else 'score_final'
    predicoes_df['fitness_score'] = predicoes_df[ranking_col]

    top = predicoes_df.nlargest(10, ranking_col)
    cols = [c for c in ['apelido', 'posicao_id', 'preco', ranking_col, 'tipo_confronto', 'score_h2h', 'selo_valorizacao']
            if c in top.columns]
    logger.info(f"\n🏆 Top 10 pós-cruzamento:\n{top[cols].to_string(index=False)}")

    # ----------------------------------------------------------
    # ETAPA 9 – SPECIALIST CHECKLIST (OBRIGATÓRIO)
    # ----------------------------------------------------------
    logger.info("\n🔎 [9/10] Executando Specialist Checklist (pré-escalação)...")
    specialist_result = {}
    try:
        from src.ml.specialist_logic import CartolaPrescalingChecklist
        checklist = CartolaPrescalingChecklist(
            rodada=rodada_atual,
            budget=PATRIMONIO,
            atletas_df=atletas_df,
            partidas_df=partidas_enriquecidas if not partidas_enriquecidas.empty else partidas_df,
            predicoes_df=predicoes_df,
            modo=args.modo,
        )
        specialist_result = checklist.run()
        cap_sugerido = specialist_result.get('recommended_captain', 'N/A')
        logger.info(f"✅ Specialist concluído | Capitão sugerido: {cap_sugerido}")
        if specialist_result.get('warnings'):
            for w in specialist_result['warnings']:
                logger.warning(f"   ⚠️  {w}")
    except Exception as e:
        logger.warning(f"⚠️  Specialist Analyzer indisponível: {e}")

    # ----------------------------------------------------------
    # ETAPA 10 – OTIMIZAÇÃO GENÉTICA → ESCALAÇÃO
    # ----------------------------------------------------------
    logger.info(f"\n🧬 [10/10] Montando melhor time...")

    opt_config = config.get('optimizer', {})
    predicoes_para_opt = predicoes_df.copy()
    if 'predicao_std' not in predicoes_para_opt.columns:
        predicoes_para_opt['predicao_std'] = 0.0

    # Incorporar odds_score se disponível
    if 'odds_score' in predicoes_para_opt.columns:
        predicoes_para_opt['fitness_score'] = (
            predicoes_para_opt['fitness_score'] * 0.8
            + predicoes_para_opt['odds_score'] * 20 * 0.2
        )

    try:
        optimizer = GeneticTeamOptimizer(
            atletas_df=atletas_df,
            predicoes=predicoes_para_opt,
            patrimonio=PATRIMONIO,
            formacao=FORMACAO or '4-3-3',
            population_size=opt_config.get('population_size', 250),
            generations=opt_config.get('generations', 150),
            mutation_rate=opt_config.get('mutation_rate', 0.20),
            elite_size=opt_config.get('elite_size', 20),
            max_mesmo_clube=opt_config.get('max_mesmo_clube', 3),
            penalidade_variancia=opt_config.get('penalidade_variancia', True)
        )
        best_team, stats = optimizer.optimize()
        team_df = optimizer.format_team_output(best_team)

        # Enriquecer output com selos + confronto
        extra_cols = [c for c in ['atleta_id', 'selo_valorizacao', 'mpv', 'margem_valorizacao',
                                   'tipo_confronto', 'score_h2h', 'score_cruzado']
                      if c in predicoes_df.columns]
        if extra_cols and 'atleta_id' in team_df.columns:
            team_df = team_df.merge(
                predicoes_df[extra_cols], on='atleta_id', how='left'
            )

        logger.info(f"\n{'=' * 70}")
        logger.info("🏆 TIME OTIMIZADO")
        logger.info(f"{'=' * 70}")
        print(team_df.to_string(index=False))
        logger.info(f"\n⚽ Pontos Preditos: {stats['total_pontos_preditos']:.2f}")
        logger.info(f"💰 Preço Total: C${stats['total_preco']:.2f} / C${PATRIMONIO:.2f}")
        if specialist_result.get('recommended_captain'):
            logger.info(f"⭐ Capitão recomendado pelo Specialist: {specialist_result['recommended_captain']}")

        output_dir = Path(args.output_dir)
        salvar_escalacao_xlsx(team_df, output_dir, rodada_atual)

    except Exception as e:
        logger.error(f"❌ Erro na otimização genética: {e}\n{traceback.format_exc()}")

    # ----------------------------------------------------------
    # ETAPA OPCIONAL – RESERVA DE LUXO
    # ----------------------------------------------------------
    if args.reserva_luxo:
        logger.info("\n🔄 [RESERVA DE LUXO] Calculando reservas ideais...")
        try:
            mega_opt = CartolaOptimizer('mega', config={
                'max_players_per_club': opt_config.get('max_mesmo_clube', 3),
                'solver_time_limit': 30,
            })
            from src.features.feature_engineering_v2 import FeatureEngineeringV2
            fe_v2 = FeatureEngineeringV2()
            df_mega = atletas_df.copy()

            if 'fitness_score' in predicoes_df.columns:
                score_map = predicoes_df.set_index('atleta_id')['fitness_score'].to_dict()
                df_mega['mega_score'] = df_mega['atleta_id'].map(score_map).fillna(0)
            else:
                df_mega = fe_v2.engineer_features(df_mega)

            for col in ['selo_valorizacao', 'mpv', 'margem_valorizacao']:
                if col in predicoes_df.columns:
                    selos = predicoes_df[['atleta_id', col]]
                    df_mega = df_mega.merge(selos, on='atleta_id', how='left')

            resultado_reserva = mega_opt.strategy.optimize_with_luxury_reserve(
                df=df_mega, budget=PATRIMONIO, formation=FORMACAO,
            )
            if resultado_reserva:
                mega_opt.strategy.print_lineup_with_reserve(resultado_reserva)
                output_dir = Path(args.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                reserva_path = output_dir / f"escalacao_reserva_luxo_r{rodada_atual}_{ts}.xlsx"
                with pd.ExcelWriter(reserva_path, engine='openpyxl') as writer:
                    resultado_reserva['titulares'].to_excel(writer, sheet_name='Titulares', index=False)
                    if not resultado_reserva['reservas'].empty:
                        resultado_reserva['reservas'].to_excel(writer, sheet_name='Reservas', index=False)
                logger.info(f"💾 Reservas salvas em: {reserva_path}")

        except Exception as e:
            logger.error(f"❌ Erro na Reserva de Luxo: {e}\n{traceback.format_exc()}")

    logger.info("\n✅ Pipeline COMPLETO! Escalação gerada com análise total da rodada.")


if __name__ == "__main__":
    main()
