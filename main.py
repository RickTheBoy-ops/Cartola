#!/usr/bin/env python3
"""
Sistema Completo de Análise e Predição do Cartola FC
Pipeline: Coleta → Feature Engineering V3 → ML → Otimização → Reserva de Luxo

Uso:
    python main.py                                 # usa .env / defaults
    python main.py --orcamento 110 --formacao 4-3-3
    python main.py --modo valorizar --reserva-luxo --v3-features
"""

import argparse
import logging
import os
from pathlib import Path

import pandas as pd
import sqlite3
import yaml
from dotenv import load_dotenv

# Carregar variáveis de ambiente
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

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ========================================================================
# HELPERS
# ========================================================================

def carregar_config(config_path: str = "config.yaml") -> dict:
    """Carrega configurações centralizadas do config.yaml"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"⚠️ {config_path} não encontrado, usando defaults")
        return {}


def parse_args() -> argparse.Namespace:
    """Parse dos argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description="Cartola FC Optimizer - Pipeline completo de predição e escalação"
    )
    parser.add_argument(
        "--orcamento",
        type=float,
        default=float(os.getenv("CARTOLA_ORCAMENTO", 110.0)),
        help="Orçamento total em cartoletas (padrão: 110)"
    )
    parser.add_argument(
        "--formacao",
        type=str,
        default=os.getenv("CARTOLA_FORMACAO", ""),
        help="Formação fixa (ex: 4-3-3). Vazio = testar todas"
    )
    parser.add_argument(
        "--modo",
        type=str,
        default=os.getenv("CARTOLA_MODO", "equilibrado"),
        choices=["equilibrado", "pontuar", "valorizar"],
        help="Modo de otimização (padrão: equilibrado)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Diretório de saída para os arquivos de escalação"
    )
    parser.add_argument(
        "--reserva-luxo",
        action="store_true",
        default=False,
        help="Incluir otimização de Reserva de Luxo (2025/2026)"
    )
    parser.add_argument(
        "--v3-features",
        action="store_true",
        default=False,
        help="Usar feature engineering V3 (MC/MF, pontos cedidos por posição, odds)"
    )
    return parser.parse_args()


# ========================================================================
# PIPELINE PRINCIPAL
# ========================================================================

def main():
    """Execução principal do pipeline"""

    args = parse_args()
    config = carregar_config()

    logger.info("=" * 60)
    logger.info("🏆 SISTEMA DE ANÁLISE CARTOLA FC")
    logger.info(f"⚙️ Orçamento: C${args.orcamento:.1f} | Modo: {args.modo}")
    logger.info(f"⚙️ Features V3: {args.v3_features} | Reserva de Luxo: {args.reserva_luxo}")
    logger.info("=" * 60)

    EMAIL = os.getenv('CARTOLA_EMAIL')
    PASSWORD = os.getenv('CARTOLA_PASSWORD')
    PATRIMONIO = args.orcamento
    FORMACAO = args.formacao if args.formacao else None

    if FORMACAO and not validar_formacao(FORMACAO):
        logger.error(f"❌ Formação '{FORMACAO}' inválida.")
        return

    # ========== 1. CLIENTE API ==========
    logger.info("\n📡 [1/6] Inicializando cliente API...")
    api_client = CartolaAPIClient(email=EMAIL, password=PASSWORD)

    # ========== 2. COLETAR DADOS ==========
    logger.info("\n📥 [2/6] Coletando dados da API...")
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

    # ========== 3. DADOS HISTÓRICOS ==========
    logger.info("\n📚 [3/6] Preparando dados históricos...")

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

    conn.close()

    # Buscar clubes direto da API (tabela 'clubes' não existe no DB local)
    try:
        import requests
        r = requests.get("https://api.cartola.globo.com/clubes",
                         headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        clubes_raw = r.json()
        clubes_df = pd.DataFrame([
            {'id': int(k), 'nome': v.get('nome', ''), 'abreviacao': v.get('abreviacao', '')}
            for k, v in clubes_raw.items()
        ])
        logger.info(f"✅ {len(clubes_df)} clubes carregados da API")
    except Exception:
        # Fallback: extrair clubes dos atletas
        clubes_df = atletas_df[['clube_id']].drop_duplicates().rename(columns={'clube_id': 'id'})
        clubes_df['nome'] = clubes_df['id'].astype(str)
        logger.warning("⚠️ Clubes via fallback (apenas IDs)")

    logger.info(f"📊 Histórico: {len(historico_df)} registros | Partidas: {len(partidas_df)}")

    # Filtrar atletas com jogo confirmado
    atletas_df = filtrar_atletas_com_jogo(atletas_df, rodada_atual, partidas_df)
    logger.info(f"✅ {len(atletas_df)} atletas com jogo confirmado na rodada {rodada_atual}")

    # ========== 4. FEATURE ENGINEERING ==========
    logger.info("\n🔧 [4/6] Criando features...")

    if len(historico_df) > 0:
        if args.v3_features:
            # V3: inclui MC/MF + pontos cedidos por posição
            historico_df = FeatureEngineer.engineer_all_features_v3(historico_df, partidas_df)
            logger.info("✅ Feature Engineering V3 aplicado (MC/MF + pontos cedidos por posição)")
        else:
            historico_df = FeatureEngineer.engineer_all_features(historico_df, partidas_df)
            logger.info("✅ Feature Engineering V2 aplicado")
    else:
        logger.warning("⚠️ Sem histórico para feature engineering.")

    # ========== 4.1. INTEGRAR ODDS (se V3 ativo) ==========
    if args.v3_features:
        logger.info("🎲 Integrando odds de probabilidade...")
        odds_integrator = OddsIntegrator()  # Chave via ODDS_API_KEY no .env
        atletas_df = odds_integrator.enrich(atletas_df, partidas_df, clubes_df)
        logger.info("✅ Odds integradas (prob_gol, prob_sg, odds_score)")

    # ========== 5. TREINAR MODELO ==========
    logger.info("\n🧠 [5/6] Treinando modelo de predição...")

    predictor = CartolaPredictor(model_type=config.get('ml', {}).get('model_type', 'rf'))

    dados_treino = historico_df[historico_df['rodada'] < rodada_atual] if len(historico_df) > 0 else pd.DataFrame()
    can_train = len(dados_treino) >= CartolaPredictor.HISTORICO_MINIMO

    if can_train:
        metrics = predictor.train(dados_treino, validate=True)
        model_path = Path("data/models/cartola_predictor.pkl")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        predictor.save_model(str(model_path))
    else:
        logger.warning(f"⚠️ Histórico insuficiente ({len(dados_treino)} registros). Usando heurística.")

    # ========== 6. PREDIÇÕES ==========
    logger.info(f"\n🎯 [6/6] Predizendo rodada {rodada_atual}...")

    if can_train and predictor.is_trained:
        ultimos_dados = historico_df.sort_values('rodada').groupby('atleta_id').tail(5)
        ultimos_agregados = ultimos_dados.sort_values('rodada').groupby('atleta_id').last().reset_index()

        dados_predicao = atletas_df.merge(
            ultimos_agregados, on='atleta_id', how='left', suffixes=('', '_hist')
        ).fillna(0)

        # Pipeline completo: predição tática + flag de valorização
        predicoes_df = predictor.predict_full(dados_predicao)
        logger.info("✅ Predições com pesos táticos e selos de valorização geradas")

    else:
        predicoes_df = CartolaPredictor.fallback_heuristica(atletas_df)
        predicoes_df['predicao_ajustada'] = predicoes_df['predicao']
        predicoes_df = CartolaPredictor.add_valorizacao_flag(predicoes_df)

    # Ajustar score por modo de otimização
    score_col = 'predicao_ajustada' if 'predicao_ajustada' in predicoes_df.columns else 'predicao'
    if args.modo == 'valorizar' and 'valorizacao_score' in predicoes_df.columns:
        predicoes_df['score_final'] = (predicoes_df[score_col] * 0.4 +
                                        predicoes_df['valorizacao_score'] * 0.6)
        logger.info("🏷️ Modo VALORIZAR: priorizando selos de valorização")
    elif args.modo == 'pontuar':
        predicoes_df['score_final'] = predicoes_df[score_col]
        logger.info("⚽ Modo PONTUAR: priorizando predição de pontos")
    else:
        # equilibrado
        val_weight = predicoes_df.get('valorizacao_score', pd.Series(50, index=predicoes_df.index))
        predicoes_df['score_final'] = (predicoes_df[score_col] * 0.65 + val_weight * 0.35)
        logger.info("⚖️ Modo EQUILIBRADO: balanceando pontos e valorização")

    logger.info(f"📊 {len(predicoes_df)} atletas com predição gerada")

    # Top 10
    top = predicoes_df.nlargest(10, 'score_final')
    cols = [c for c in ['apelido', 'posicao_id', 'preco', 'score_final', 'selo_valorizacao'] if c in top.columns]
    logger.info(f"\n🏆 Top 10:\n{top[cols].to_string(index=False)}")

    # ========== 7. OTIMIZAÇÃO PRINCIPAL ==========
    logger.info(f"\n🧬 [OTIMIZAÇÃO] Montando melhor time...")

    opt_config = config.get('optimizer', {})

    predicoes_para_opt = predicoes_df.copy()
    if 'predicao_std' not in predicoes_para_opt.columns:
        predicoes_para_opt['predicao_std'] = 0.0
    predicoes_para_opt['fitness_score'] = predicoes_para_opt['score_final']

    # Adicionar odds_score como coluna extra se disponível (para o otimizador usar)
    if 'odds_score' in predicoes_para_opt.columns:
        predicoes_para_opt['fitness_score'] = (
            predicoes_para_opt['fitness_score'] * 0.8 +
            predicoes_para_opt['odds_score'] * 20 * 0.2
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

        # Adicionar selos de valorização ao output
        if 'selo_valorizacao' in predicoes_df.columns and 'atleta_id' in team_df.columns:
            selos = predicoes_df[['atleta_id', 'selo_valorizacao', 'mpv', 'margem_valorizacao']].copy()
            team_df = team_df.merge(selos, on='atleta_id', how='left')

        logger.info(f"\n{'=' * 70}")
        logger.info("🏆 TIME OTIMIZADO")
        logger.info(f"{'=' * 70}")
        print(team_df.to_string(index=False))
        logger.info(f"\n⚽ Pontos Preditos: {stats['total_pontos_preditos']:.2f}")
        logger.info(f"💰 Preço Total: C${stats['total_preco']:.2f} / C${PATRIMONIO:.2f}")

        # Salvar output
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        xlsx_path = output_dir / f"escalacao_rodada{rodada_atual}_{ts}.xlsx"
        team_df.to_excel(xlsx_path, index=False)
        logger.info(f"💾 Escalação salva em: {xlsx_path}")

    except Exception as e:
        import traceback
        logger.error(f"❌ Erro na otimização genética: {e}\n{traceback.format_exc()}")

    # ========== 8. RESERVA DE LUXO (opcional) ==========
    if args.reserva_luxo:
        logger.info("\n🔄 [RESERVA DE LUXO] Calculando reservas ideais...")
        try:
            mega_opt = CartolaOptimizer('mega', config={
                'max_players_per_club': opt_config.get('max_mesmo_clube', 3),
                'solver_time_limit': 30,
            })

            # Preparar df para o solver PuLP (precisa de mega_score)
            from src.features.feature_engineering_v2 import FeatureEngineeringV2
            fe_v2 = FeatureEngineeringV2()
            df_mega = atletas_df.copy()
            if 'score_final' in predicoes_df.columns:
                score_map = predicoes_df.set_index('atleta_id')['score_final'].to_dict()
                df_mega['mega_score'] = df_mega['atleta_id'].map(score_map).fillna(0)
            else:
                df_mega = fe_v2.engineer_features(df_mega)

            # Adicionar selos de valorização ao df do mega optimizer
            if 'selo_valorizacao' in predicoes_df.columns:
                selos = predicoes_df[['atleta_id', 'selo_valorizacao', 'mpv', 'margem_valorizacao']].copy()
                df_mega = df_mega.merge(selos, on='atleta_id', how='left')

            resultado_reserva = mega_opt.strategy.optimize_with_luxury_reserve(
                df=df_mega,
                budget=PATRIMONIO,
                formation=FORMACAO,
            )

            if resultado_reserva:
                mega_opt.strategy.print_lineup_with_reserve(resultado_reserva)

                # Salvar escalação com reservas
                output_dir = Path(args.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")

                reserva_path = output_dir / f"escalacao_reserva_luxo_r{rodada_atual}_{ts}.xlsx"
                with pd.ExcelWriter(reserva_path, engine='openpyxl') as writer:
                    resultado_reserva['titulares'].to_excel(writer, sheet_name='Titulares', index=False)
                    if not resultado_reserva['reservas'].empty:
                        resultado_reserva['reservas'].to_excel(writer, sheet_name='Reservas', index=False)
                logger.info(f"💾 Escalação com reservas salva em: {reserva_path}")

        except Exception as e:
            import traceback
            logger.error(f"❌ Erro na Reserva de Luxo: {e}\n{traceback.format_exc()}")

    logger.info("\n✅ Pipeline concluído!")


if __name__ == "__main__":
    main()
