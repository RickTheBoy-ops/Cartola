#!/usr/bin/env python3
"""
Sistema Completo de Análise e Predição do Cartola FC
"""

import logging
from pathlib import Path
import pandas as pd
import sqlite3
import os
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Imports do projeto
from src.api.client import CartolaAPIClient
from src.data.collector import CartolaDataCollector
from src.ml.features import FeatureEngineer
from src.ml.predictor import CartolaPredictor
from src.ml.optimizer import GeneticTeamOptimizer

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Execução principal"""
    
    # ========== 1. CONFIGURAÇÃO ==========
    logger.info("="*60)
    logger.info("SISTEMA DE ANÁLISE CARTOLA FC")
    logger.info("="*60)
    
    # Credenciais via .env ou manual
    EMAIL = os.getenv('CARTOLA_EMAIL')
    PASSWORD = os.getenv('CARTOLA_PASSWORD')
    
    # Seu patrimônio atual no Cartola
    PATRIMONIO = float(os.getenv('CARTOLA_PATRIMONIO', 100.0))
    
    # Formação desejada
    FORMACAO = os.getenv('CARTOLA_FORMACAO', '4-3-3')
    
    # ========== 2. INICIALIZAR CLIENTE API ==========
    logger.info("\n[1/6] Inicializando cliente API...")
    api_client = CartolaAPIClient(email=EMAIL, password=PASSWORD)
    
    # ========== 3. COLETAR DADOS ==========
    logger.info("\n[2/6] Coletando dados da API...")
    collector = CartolaDataCollector(api_client)
    
    # Obter rodada atual
    try:
        mercado = collector.collect_mercado_status()
        rodada_atual = mercado['rodada_atual']
        logger.info(f"Rodada Atual: {rodada_atual}")
        
        # Coletar atletas do mercado
        atletas_df = collector.collect_atletas_mercado(rodada_atual)
        logger.info(f"Coletados {len(atletas_df)} atletas")
        
        # Coletar partidas
        collector.collect_partidas(rodada_atual)
    except Exception as e:
        logger.error(f"Erro na coleta de dados: {e}")
        return

    # ========== 4. PREPARAR DADOS HISTÓRICOS ==========
    logger.info("\n[3/6] Preparando dados históricos...")
    
    # Carregar histórico do banco
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
    
    logger.info(f"Histórico carregado: {len(historico_df)} registros")
    
    # ========== 5. FEATURE ENGINEERING ==========
    logger.info("\n[4/6] Criando features...")
    
    if len(historico_df) > 0:
        historico_df = FeatureEngineer.engineer_all_features(historico_df, partidas_df)
        logger.info(f"Features criadas: {historico_df.shape[1]} colunas")
    else:
        logger.warning("Sem histórico suficiente para predições baseadas em ML.")
    
    # ========== 6. TREINAR MODELO ==========
    logger.info("\n[5/6] Treinando modelo de predição...")
    
    predictor = CartolaPredictor(model_type='rf')
    
    # Se não houver histórico suficiente no banco local, vamos usar uma abordagem simplificada
    # ou tentar baixar dados históricos (se existissem endpoints para isso no MD)
    
    can_train = len(historico_df[historico_df['rodada'] < rodada_atual]) > 50
    
    if can_train:
        train_df = historico_df[historico_df['rodada'] < rodada_atual].copy()
        metrics = predictor.train(train_df, validate=True)
        
        model_path = Path("data/models/cartola_predictor.pkl")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        predictor.save_model(str(model_path))
    else:
        logger.warning("Histórico insuficiente para treino de ML. Usando heurística de média.")
        # Simula predição baseada na média atual para atletas da rodada
        predicoes_df = atletas_df[['atleta_id', 'apelido', 'posicao_id', 'clube_id', 'preco', 'media']].copy()
        predicoes_df['predicao'] = predicoes_df['media']
    
    # ========== 7. PREDIZER PRÓXIMA RODADA ==========
    logger.info(f"\n[6/6] Predizendo rodada {rodada_atual}...")
    
    if can_train:
        # Pegar dados mais recentes de cada atleta para predizer a rodada atual
        ultimos_dados = historico_df.sort_values('rodada').groupby('atleta_id').tail(1)
        
        # Merge com o mercado atual para garantir preços e status atualizados
        dados_predicao = atletas_df.merge(
            ultimos_dados,
            on='atleta_id',
            how='left',
            suffixes=('', '_hist')
        )
        
        # Preencher NaNs para novos atletas
        dados_predicao = dados_predicao.fillna(0)
        
        predicoes_df = predictor.predict_with_confidence(dados_predicao)
    else:
        # predicoes_df já foi criado na heurística acima
        pass
        
    logger.info(f"Predições geradas para {len(predicoes_df)} atletas")
    
    # Top 10 predições
    top_predicoes = predicoes_df.nlargest(10, 'predicao')
    logger.info(f"\nTop 10 Predições:\n{top_predicoes[['apelido', 'posicao_id', 'preco', 'predicao']]}")
    
    # ========== 8. OTIMIZAR ESCALAÇÃO ==========
    logger.info(f"\n[OTIMIZAÇÃO] Montando melhor time para formação {FORMACAO}...")
    
    try:
        optimizer = GeneticTeamOptimizer(
            atletas_df=atletas_df,
            predicoes=predicoes_df,
            patrimonio=PATRIMONIO,
            formacao=FORMACAO,
            population_size=150,
            generations=100,
            mutation_rate=0.15
        )
        
        best_team, stats = optimizer.optimize()
        
        # Exibir time otimizado
        team_df = optimizer.format_team_output(best_team)
        
        logger.info(f"\n{'='*70}")
        logger.info("TIME OTIMIZADO")
        logger.info(f"{'='*70}")
        print(team_df.to_string(index=False))
        logger.info(f"\n{'='*70}")
        logger.info(f"Pontos Totais Preditos: {stats['total_pontos_preditos']:.2f}")
        logger.info(f"Preço Total: C$ {stats['total_preco']:.2f} / C$ {PATRIMONIO:.2f}")
        logger.info(f"Patrimônio Usado: {stats['patrimonio_usado']:.1f}%")
        logger.info(f"{'='*70}\n")
        
        # Salvar resultado
        output_path = Path("data/processed/time_sugerido.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        team_df.to_csv(output_path, index=False)
        logger.info(f"Time salvo em: {output_path}")
    except Exception as e:
        import traceback
        logger.error(f"Erro na otimização: {e}\n{traceback.format_exc()}")
    
    logger.info("\n✅ Execução concluída!")

if __name__ == "__main__":
    main()
