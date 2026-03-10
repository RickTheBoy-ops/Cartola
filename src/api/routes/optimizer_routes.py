from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict

# Assumindo que a factory e dados mock/raw possam ser passados ou carregados
from src.optimizer.factory import CartolaOptimizer
import pandas as pd
import sqlite3
import os
from pathlib import Path

router = APIRouter()

class OptimizationRequest(BaseModel):
    strategy: str = "mega"
    budget: float = 100.0
    formation: Optional[str] = "4-3-3"
    config: Optional[Dict] = {}

class OptimizationResponse(BaseModel):
    total_cost: float
    total_score: float
    captain: str
    players: list

@router.post("/optimize", response_model=OptimizationResponse)
def optimize_team(request: OptimizationRequest):
    try:
        # Puxaremos do DB por conveniência, a menos que especificado
        db_path = Path(__file__).parent.parent.parent.parent / "data" / "cartola.db"
        
        # Para demonstração na API, carregamos os últimos atletas da rodada:
        # Se n houver DB viável, levantaremos erro.
        if not db_path.exists():
            raise HTTPException(status_code=404, detail="Banco de dados não encontrado localmente para gerar df.")
            
        conn = sqlite3.connect(db_path)
        # Buscar algo básico. Em prod chamaríamos o collector.
        query = "SELECT * FROM atletas LIMIT 50" # Simplificado só para o mock responder rápido
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Gerar algumas mock predictions (já que é apenas um schema mock de fallback)
        if 'mega_score' not in df.columns:
            import numpy as np
            df['mega_score'] = np.random.uniform(2, 12, size=len(df))
            df['preco'] = np.random.uniform(5, 15, size=len(df))
            df['clube_id'] = np.random.randint(1, 20, size=len(df))
            df['posicao_id'] = np.random.choice([1, 2, 3, 4, 5, 6], size=len(df))
        
        optimizer = CartolaOptimizer(strategy=request.strategy, config=request.config)
        lineup = optimizer.optimize(df, budget=request.budget, formation=request.formation)
        
        if lineup is None or len(lineup) < 12:
            raise HTTPException(status_code=400, detail="Não foi possível gerar um time válido.")
            
        captain = optimizer.select_captain(lineup)
        
        players_list = []
        for _, row in lineup.iterrows():
            players_list.append({
                "atleta_id": int(row.get("atleta_id", 0)),
                "apelido": row.get("apelido", "Desconhecido"),
                "posicao_id": int(row.get("posicao_id", 0)),
                "preco": float(row.get("preco", 0.0)),
                "score_projetado": float(row.get("mega_score", row.get("predicao", 0.0)))
            })
            
        return OptimizationResponse(
            total_cost=float(lineup['preco'].sum()),
            total_score=float(optimizer.calculate_score(lineup)),
            captain=captain,
            players=players_list
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
