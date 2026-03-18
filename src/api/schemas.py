from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any

class AtletaModel(BaseModel):
    atleta_id: int
    apelido: str
    preco: float
    clube_id: int
    posicao_id: int
    status_id: int = 7
    mega_score: Optional[float] = 0.0
    media: Optional[float] = 0.0

    @validator('preco')
    def preco_valido(cls, v):
        if v < 0:
            raise ValueError('Preço deve ser positivo')
        return v

    @validator('posicao_id')
    def posicao_valida(cls, v):
        if v not in [1, 2, 3, 4, 5, 6]:
            raise ValueError('Posição deve ser entre 1 e 6 (GOL, LAT, ZAG, MEI, ATA, TEC)')
        return v

class OptimizationRequest(BaseModel):
    strategy: str = Field(default="genetic", description="Estratégia de otimização (genetic, mega)")
    budget: float = Field(default=110.0, description="Orçamento disponível em cartoletas")
    formation: Optional[str] = Field(default="4-3-3", description="Formação tática. Ex: 4-3-3, 3-4-3")
    config: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Configurações (modo: pontuar/valorizar/equilibrado)")

    @validator('budget')
    def budget_positivo(cls, v):
        if v <= 0:
            raise ValueError('Orçamento (budget) deve ser maior que zero')
        return v

class PlayerResponse(BaseModel):
    apelido: str
    posicao_nome: str
    clube: str
    preco: float
    score_projetado: float
    pontos_esperados: float
    minimo_para_valorizar: float

class OptimizationResponse(BaseModel):
    total_cost: float
    total_score: float
    captain: str
    players: List[PlayerResponse]
