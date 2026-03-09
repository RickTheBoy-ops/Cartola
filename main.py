#!/usr/bin/env python3
"""
Sistema Completo de Análise e Predição do Cartola FC
Pipeline: Coleta → Feature Engineering → ML → Otimização Genética
"""

import logging
from pathlib import Path
import pandas as pd
import sqlite3
import os
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
from src.utils.validators import validar_mercado, validar_formacao, filtrar_atletas_com_jogo

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def carregar_config(config_path: str = "config.yaml") -> dict:
    """Carrega configurações centralizadas do config.yaml"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"⚠️ {config_path} não encontrado, usando defaults")
        return {}


def main():
    """Execução principal do pipeline"""

    config = carregar_config()

    # ========== 1. CONFIGURAÇÃO ==========
    logger.info("=" * 60)
    logger.info("🏆 SISTEMA DE ANÁLISE CARTOLA FC")
    logger.info("=" * 60)

    EMAIL = os.getenv('CARTOLA_EMAIL')
    PASSWORD = os.getenv('CARTOLA_PASSWORD')
    PATRIMONIO = float(os.getenv('CARTOLA_PATRIMONIO', 100.0))
    FORMACAO = os.getenv('CARTOLA_FORMACAO', '4-3-3')

    # Validar formação
    if not validar_formacao(FORMACAO):
        logger.error(f"❌ Formação '{FORMACAO}' inválida. Use: 3-4-3, 3-5-2, 4-3-3, 4-4-2, 4-5-1, 5-3-2, 5-4-1")
        return

    logger.info(f"⚙️ Patrimônio: C$ {PATRIMONIO:.2f} | Formação: {FORMACAO}")

    # ========== 2. INICIALIZAR CLIENTE API ==========
    logger.info("\n📡 [1/6] Inicializando cliente API...")
    api_client = CartolaAPIClient(email=EMAIL, password=PASSWORD)

    # ========== 3. COLETAR DADOS ==========
    logger.info("\n📥 [2/6] Coletando dados da API...")
    collector = CartolaDataCollector(api_client)

    try:
        mercado = collector.collect_mercado_status()

        # Validar mercado
        mercado_info = validar_mercado(mercado)
        if not mercado_info['valido']:
            logger.error(f"❌ {mercado_info['mensagem']}")
            return

        rodada_atual = mercado_info['rodada_atual']
        logger.info(f"📊 {mercado_info['mensagem']}")

        # Coletar atletas do mercado
        atletas_df = collector.collect_atletas_mercado(rodada_atual)
        logger.info(f"✅ Coletados {len(atletas_df)} atletas")

        if len(atletas_df) == 0:
            logger.error("❌ Nenhum atleta encontrado. Verifique a conexão com a API.")
            return

        # Coletar partidas
        collector.collect_partidas(rodada_atual)
    except Exception as e:
        logger.error(f"❌ Erro na coleta de dados: {e}")
        return

    # ========== 3.5. FILTRAR ATLETAS SEM JOGO NA RODADA ==========
    if len(atletas_df) > 0 and len(partidas_df if 'partidas_df' in dir() else []) == 0:
        # partidas_df ainda não foi carregado aqui; filtro aplicado mais abaixo
        pass

    # ========== 4. PREPARAR DADOS HISTÓRICOS ==========
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

    logger.info(f"📊 Histórico carregado: {len(historico_df)} registros de pontuação")

    # Filtrar atletas sem jogo confirmado na rodada atual
    atletas_df = filtrar_atletas_com_jogo(atletas_df, rodada_atual, partidas_df)
    logger.info(f"✅ {len(atletas_df)} atletas com jogo confirmado na rodada {rodada_atual}")

    # ========== 5. FEATURE ENGINEERING ==========
    logger.info("\n🔧 [4/6] Criando features...")

    if len(historico_df) > 0:
        historico_df = FeatureEngineer.engineer_all_features(historico_df, partidas_df)
    else:
        logger.warning("⚠️ Sem histórico para feature engineering.")

    # ========== 6. TREINAR MODELO ==========
    logger.info("\n🧠 [5/6] Treinando modelo de predição...")

    predictor = CartolaPredictor(model_type=config.get('ml', {}).get('model_type', 'rf'))

    # Verificar se há dados suficientes para ML
    dados_treino = historico_df[historico_df['rodada'] < rodada_atual] if len(historico_df) > 0 else pd.DataFrame()
    can_train = len(dados_treino) >= CartolaPredictor.HISTORICO_MINIMO

    if can_train:
        train_df = dados_treino.copy()
        metrics = predictor.train(train_df, validate=True)

        model_path = Path("data/models/cartola_predictor.pkl")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        predictor.save_model(str(model_path))
    else:
        logger.warning(f"⚠️ Histórico insuficiente ({len(dados_treino)} registros). Usando heurística.")

    # ========== 7. PREDIZER PRÓXIMA RODADA ==========
    logger.info(f"\n🎯 [6/6] Predizendo rodada {rodada_atual}...")

    if can_train and predictor.is_trained:
        # Pegar últimas 5 rodadas de cada atleta (não apenas 1)
        ultimos_dados = historico_df.sort_values('rodada').groupby('atleta_id').tail(5)

        # Calcular média ponderada exponencial (rodadas mais recentes têm mais peso)
        ultimos_dados = ultimos_dados.copy()
        ultimos_dados['peso_rodada'] = ultimos_dados.groupby('atleta_id').cumcount() + 1
        ultimos_dados['pontos_ewm_local'] = (
            ultimos_dados.groupby('atleta_id')['pontos']
            .transform(lambda x: x.ewm(span=3, min_periods=1).mean())
        )

        # Agregar: pegar o último estado de cada atleta (com features já calculadas)
        ultimos_agregados = ultimos_dados.sort_values('rodada').groupby('atleta_id').last().reset_index()

        dados_predicao = atletas_df.merge(
            ultimos_agregados,
            on='atleta_id',
            how='left',
            suffixes=('', '_hist')
        )
        dados_predicao = dados_predicao.fillna(0)

        # Usar pesos táticos se as novas features estiverem disponíveis
        tem_features_taticas = (
            'bonus_oponente_fraco' in dados_predicao.columns and
            'mando_casa' in dados_predicao.columns
        )

        if tem_features_taticas:
            predicoes_df = predictor.predict_with_tactical_weights(dados_predicao)
            logger.info("✅ Predições com pesos táticos aplicados")
        else:
            predicoes_df = predictor.predict_with_confidence(dados_predicao)
            # Fallback: predicao_ajustada = predicao base
            predicoes_df['predicao_ajustada'] = predicoes_df['predicao']
            logger.info("⚠️ Features táticas indisponíveis — usando predição base")
    else:
        # Fallback heurístico (média ponderada)
        predicoes_df = CartolaPredictor.fallback_heuristica(atletas_df)
        predicoes_df['predicao_ajustada'] = predicoes_df['predicao']

    logger.info(f"📊 Predições geradas para {len(predicoes_df)} atletas")

    # Top 10 predições (usando predicao_ajustada se disponível)
    score_col = 'predicao_ajustada' if 'predicao_ajustada' in predicoes_df.columns else 'predicao'
    top_predicoes = predicoes_df.nlargest(10, score_col)
    cols_show = [c for c in ['apelido', 'posicao_id', 'preco', 'predicao', 'predicao_ajustada'] if c in top_predicoes.columns]
    logger.info(f"\n🏆 Top 10 Predições:\n{top_predicoes[cols_show].to_string(index=False)}")

    # ========== 8. OTIMIZAR ESCALAÇÃO ==========
    logger.info(f"\n🧬 [OTIMIZAÇÃO] Montando melhor time para formação {FORMACAO}...")

    opt_config = config.get('optimizer', {})

    # Merge predicao_std para o otimizador (se disponível, para penalidade de variância)
    predicoes_para_opt = predicoes_df.copy()
    if 'predicao_std' not in predicoes_para_opt.columns:
        predicoes_para_opt['predicao_std'] = 0.0

    # Usar predicao_ajustada como score principal do otimizador
    if 'predicao_ajustada' in predicoes_para_opt.columns:
        predicoes_para_opt['predicao'] = predicoes_para_opt['predicao_ajustada']

    try:
        optimizer = GeneticTeamOptimizer(
            atletas_df=atletas_df,
            predicoes=predicoes_para_opt,
            patrimonio=PATRIMONIO,
            formacao=FORMACAO,
            population_size=opt_config.get('population_size', 250),
            generations=opt_config.get('generations', 150),
            mutation_rate=opt_config.get('mutation_rate', 0.20),
            elite_size=opt_config.get('elite_size', 20),
            max_mesmo_clube=opt_config.get('max_mesmo_clube', 3),
            penalidade_variancia=opt_config.get('penalidade_variancia', True)
        )

        best_team, stats = optimizer.optimize()

        # Exibir time otimizado
        team_df = optimizer.format_team_output(best_team)

        logger.info(f"\n{'=' * 70}")
        logger.info("🏆 TIME OTIMIZADO")
        logger.info(f"{'=' * 70}")
        print(team_df.to_string(index=False))
        logger.info(f"\n{'=' * 70}")
        logger.info(f"⚽ Pontos Totais Preditos: {stats['total_pontos_preditos']:.2f}")
        logger.info(f"💰 Preço Total: C$ {stats['total_preco']:.2f} / C$ {PATRIMONIO:.2f}")
        logger.info(f"📊 Patrimônio Usado: {stats['patrimonio_usado']:.1f}%")
        logger.info(f"{'=' * 70}\n")

        # Salvar resultado
        output_path = Path("data/processed/time_sugerido.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        team_df.to_csv(output_path, index=False)
        logger.info(f"💾 Time salvo em: {output_path}")
    except Exception as e:
        import traceback
        logger.error(f"❌ Erro na otimização: {e}\n{traceback.format_exc()}")

    logger.info("\n✅ Execução concluída!")


if __name__ == "__main__":
    main()
