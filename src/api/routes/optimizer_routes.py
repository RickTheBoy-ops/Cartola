from fastapi import APIRouter, HTTPException
import os
import pandas as pd
import sqlite3
from pathlib import Path
from dotenv import load_dotenv

from src.api.client import CartolaAPIClient
from src.data.collector import CartolaDataCollector
from src.ml.features import FeatureEngineer
from src.ml.predictor import CartolaPredictor
from src.ml.optimizer import GeneticTeamOptimizer
from src.utils.validators import validar_mercado, filtrar_atletas_com_jogo
from src.api.schemas import OptimizationRequest, OptimizationResponse, PlayerResponse

router = APIRouter()

# Carregar variáveis locais se existirem
load_dotenv()

@router.post("/optimize", response_model=OptimizationResponse)
def optimize_team(request: OptimizationRequest):
    try:
        EMAIL = os.getenv('CARTOLA_EMAIL')
        PASSWORD = os.getenv('CARTOLA_PASSWORD')

        # 1. API & COLETOR
        api_client = CartolaAPIClient(email=EMAIL, password=PASSWORD)
        collector = CartolaDataCollector(api_client)

        try:
            mercado = collector.collect_mercado_status()
            mercado_info = validar_mercado(mercado)
            if not mercado_info['valido']:
                raise ValueError(mercado_info['mensagem'])
        except Exception as e:
            # Fallback para poder rodar offline
            mercado_info = {'rodada_atual': 7}
            
        rodada_atual = mercado_info['rodada_atual']
        
        try:
            atletas_df = collector.collect_atletas_mercado(rodada_atual)
            if len(atletas_df) == 0:
                raise ValueError("Nenhum atleta retornado pela API da rodada atual.")
            collector.collect_partidas(rodada_atual)
        except Exception as e:
            # Fallback local se a API estiver fora
            db_path = Path(__file__).parent.parent.parent.parent / "data" / "cartola.db"
            if not db_path.exists():
                raise HTTPException(status_code=500, detail="Sem acesso a DB local ou API na nuvem.")
            conn = sqlite3.connect(db_path)
            atletas_df = pd.read_sql_query("SELECT * FROM atletas", conn)
            # Simplificação: usar tudo
            conn.close()

        # 2. BANDO DE DADOS / HISTÓRICO
        # Se Db path n existir, instanciar db offline
        historico_df = pd.DataFrame()
        partidas_df = pd.DataFrame()
        db_path = Path(collector.db_path)
        
        if db_path.exists():
            conn = sqlite3.connect(db_path)
            # tenta puxar
            try:
                historico_df = pd.read_sql_query(f"SELECT p.*, a.clube_id FROM pontuacoes p JOIN atletas a ON p.atleta_id = a.atleta_id WHERE p.rodada >= {max(1, rodada_atual - 15)} AND p.rodada <= {rodada_atual}", conn)
                partidas_df = pd.read_sql_query(f"SELECT * FROM partidas WHERE rodada >= {max(1, rodada_atual - 15)} AND rodada <= {rodada_atual}", conn)
                atletas_df = filtrar_atletas_com_jogo(atletas_df, rodada_atual, partidas_df)
            except:
                pass
            conn.close()

        # 3. FEATURE ENGINEERING
        if len(historico_df) > 0 and len(partidas_df) > 0:
            # Tenta eng de features
            historico_df = FeatureEngineer.engineer_all_features(historico_df, partidas_df)
            
        # 4. PREDICT
        predictor = CartolaPredictor(model_type='rf')
        dados_treino = historico_df[historico_df['rodada'] < rodada_atual] if len(historico_df) > 0 else pd.DataFrame()
        
        if len(dados_treino) >= CartolaPredictor.HISTORICO_MINIMO:
            predictor.train(dados_treino, validate=False)
            ultimos_dados = historico_df.sort_values('rodada').groupby('atleta_id').tail(5).groupby('atleta_id').last().reset_index()
            dados_predicao = atletas_df.merge(ultimos_dados, on='atleta_id', how='left', suffixes=('', '_hist')).fillna(0)
            predicoes_df = predictor.predict_full(dados_predicao)
        else:
            # Fallback heuristico
            predicoes_df = CartolaPredictor.fallback_heuristica(atletas_df)
            predicoes_df['predicao_ajustada'] = predicoes_df['predicao']
            predicoes_df = CartolaPredictor.add_valorizacao_flag(predicoes_df)

        # Tratar Modo / Config
        score_col = 'predicao_ajustada' if 'predicao_ajustada' in predicoes_df.columns else 'predicao'
        modo = request.config.get("modo", "equilibrado")
        
        if modo == 'valorizar' and 'valorizacao_score' in predicoes_df.columns:
            predicoes_df['score_final'] = (predicoes_df[score_col] * 0.4 + predicoes_df['valorizacao_score'] * 0.6)
        elif modo == 'pontuar':
            predicoes_df['score_final'] = predicoes_df[score_col]
        else: # equilibrado
            val_weight = predicoes_df.get('valorizacao_score', pd.Series(50, index=predicoes_df.index))
            predicoes_df['score_final'] = (predicoes_df[score_col] * 0.65 + val_weight * 0.35)

        # 5. OTIMIZAR TIME
        predicoes_para_opt = predicoes_df.copy()
        if 'predicao_std' not in predicoes_para_opt.columns:
            predicoes_para_opt['predicao_std'] = 0.0
        
        # Mapeamos o score para usar no nosso novo fitness_score
        predicoes_para_opt['fitness_score'] = predicoes_para_opt['score_final']

        optimizer = GeneticTeamOptimizer(
            atletas_df=atletas_df,
            predicoes=predicoes_para_opt,
            patrimonio=request.budget,
            formacao=request.formation or '4-3-3',
            population_size=150, # Fast request mode
            generations=80
        )
        
        best_team, stats = optimizer.optimize()
        team_df = optimizer.format_team_output(best_team)

        players_list = []
        captain_name = ""
        
        for _, row in team_df.iterrows():
            nome = str(row.get("Atleta", "Desconhecido"))
            if "(C)" in nome:
                captain_name = nome

            # Lidar robustamente com o retorno
            score_proj = float(row.get("Score Otimização", row.get("Pred (Pts)", row.get("Predição", 0))))
            pontos_esp = float(row.get("Pred (Pts)", row.get("Predição", score_proj)))

            players_list.append({
                "clube": str(row.get("Clube", "")),
                "apelido": nome,
                "posicao_nome": str(row.get("Posição", "")),
                "preco": float(row.get("Preço (C$)", 0.0)),
                "score_projetado": round(score_proj, 2),
                "pontos_esperados": round(pontos_esp, 2),
                "minimo_para_valorizar": round(float(row.get("MPV", 0.0)), 2)
            })

        return OptimizationResponse(
            total_cost=round(float(stats.get('total_preco', 0)), 2),
            total_score=round(float(stats.get('total_pontos_preditos', 0)), 2),
            captain=captain_name,
            players=players_list
        )

    except Exception as e:
        import traceback
        error_info = traceback.format_exc()
        print(error_info)
        raise HTTPException(status_code=500, detail=f"Erro interno no Pipeline M.L.: {str(e)}")
