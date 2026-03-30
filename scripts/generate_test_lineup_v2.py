import sys
import os
import pandas as pd
import sqlite3
import logging
import warnings

# Silenciar avisos de pandas e statsmodels
warnings.filterwarnings('ignore')

# Configurar logs
logging.basicConfig(level=logging.ERROR) # Mudar para INFO se quiser ver detalhes

# Adicionar a raiz do projeto (c:\RASPAGEM CARTOLA\Cartola) ao path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.models.team_ranking import PoissonTeamRanking
    from src.ml.features import FeatureEngineer
    from src.ml.optimizer import GeneticTeamOptimizer
    from src.ml.predictor import CartolaPredictor
except ImportError as e:
    print(f"❌ Erro de importação: {e}")
    # Mostrar o path atual para debug em caso de erro
    # print(f"DEBUG: sys.path = {sys.path}")
    sys.exit(1)

def main():
    print("="*60)
    print("🏟️  GERADOR DE ESCALAÇÃO TESTE (V2.2 - NOVO NÚCLEO)")
    print("="*60)

    db_path = os.path.join("data", "cartola.db")
    if not os.path.exists(db_path):
        print(f"❌ Banco de dados não encontrado em {db_path}")
        return

    conn = sqlite3.connect(db_path)
    
    # 1. Carregar dados de 2024 rodada 38 (última rodada do histórico)
    # Vamos usar essa rodada como se fosse a 'rodada atual' para teste
    print("\n📂 Carregando dados da Rodada 38 (2024)...")
    
    df_hist = pd.read_sql_query("SELECT * FROM pontuacoes WHERE ano = 2024 AND rodada = 38", conn)
    df_atletas = pd.read_sql_query("SELECT * FROM atletas", conn)
    # Partidas para o Poisson (usamos rodadas anteriores para treinar)
    df_partidas = pd.read_sql_query("SELECT * FROM partidas WHERE ano = 2024 AND rodada < 38", conn)
    
    conn.close()

    if df_hist.empty:
        print("❌ Dados da Rodada 38 não encontrados no DB histórico.")
        return

    # 2. Treinar Poisson Team Ranking
    print("📈 Treinando Poisson Team Ranking com rodadas anteriores...")
    poisson = PoissonTeamRanking()
    try:
        poisson.fit(df_partidas)
        p_sg = poisson.predict_clean_sheet_prob(1, 2) # Exemplo
        print(f"   ✅ Poisson treinado. Exemplo P(SG): {p_sg:.2%}")
    except Exception as e:
        print(f"   ⚠️ Poisson falhou (dados insuficientes?): {e}")

    # 3. Feature Engineering (v3) - score_sem_sg
    print("🧠 Calculando novas features (score_sem_sg)...")
    df_current = df_hist.merge(df_atletas[['atleta_id', 'apelido', 'clube_id', 'posicao_id']], on='atleta_id', how='left')
    df_current = FeatureEngineer.add_score_no_cleansheets(df_current)
    
    # 4. Simulação de Predição (Mock)
    # Como não temos um modelo .pkl treinado, usamos 'pontos' ou 'media' como predicao base
    df_current['predicao'] = df_current['score_sem_sg'].fillna(0).clip(lower=0)
    df_current['predicao_std'] = 1.0
    
    # 5. Otimização
    print("🚀 Iniciando Otimizador Genético (Fitness com Capitão)...")
    
    # Filtro de status (7 = Provável no Cartola)
    # No histórico nem sempre temos o status do dia, então pegamos os top por pontos
    df_ready = df_current.sort_values('predicao', ascending=False).head(150)
    
    optimizer = GeneticTeamOptimizer(
        patrimonio=120.0,
        max_mesmo_clube=3,
        strategy='genetic'
    )
    
    # O otimizador espera uma lista de dicionários
    atletas_pool = df_ready.to_dict('records')
    
    # Executar AG
    team, metrics = optimizer.optimize(atletas_pool)
    
    # 6. Mostrar Resultado
    if team:
        print("\n" + "🏆" + " " + "TIME ESCALADO (TESTE)" + " " + "🏆")
        print("-" * 65)
        # Formatar saída (já corrigi o double-captain no optimizer.py)
        output_df = optimizer.format_team_output(team)
        
        # Mapear IDs de posição para nomes
        pos_map = {1: 'GOL', 2: 'LAT', 3: 'ZAG', 4: 'MEI', 5: 'ATA', 6: 'TEC'}
        output_df['posicao'] = output_df['posicao_id'].map(pos_map)
        
        # Exibir colunas principais
        cols = ['apelido', 'posicao', 'preco', 'predicao']
        print(output_df[cols].to_string(index=False))
        
        print("-" * 65)
        print(f"💰 Custo Total:   {metrics['cost']:.2f} / 120.00")
        print(f"🎯 Score Projetado: {metrics['score']:.2f}")
        print(f"🏺 Nota: O capitão (C) tem score dobrado no total.")
        print("="*60)
    else:
        print("\n❌ Nenhuma escalação viável encontrada.")

if __name__ == "__main__":
    main()
