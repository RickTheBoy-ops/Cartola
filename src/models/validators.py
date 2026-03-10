"""Modelos de validação usando Pydantic para garantir integridade dos dados."""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator, root_validator
from datetime import datetime


class AtletaModel(BaseModel):
    """Modelo de validação para dados de atleta."""
    
    atleta_id: int = Field(..., gt=0, description="ID único do atleta")
    apelido: str = Field(..., min_length=1, max_length=100, description="Nome/apelido do atleta")
    clube_id: int = Field(..., gt=0, description="ID do clube")
    posicao_id: int = Field(..., ge=1, le=6, description="ID da posição (1-6)")
    preco: float = Field(..., gt=0, description="Preço em cartoletas")
    pontos_rodada: Optional[float] = Field(None, description="Pontos na última rodada")
    media_pontos: Optional[float] = Field(None, ge=0, description="Média de pontos")
    jogos: Optional[int] = Field(None, ge=0, description="Total de jogos")
    status_id: Optional[int] = Field(None, description="Status do atleta (7=contundido, etc)")
    
    @validator('preco')
    def preco_valido(cls, v):
        """Valida se o preço está em range aceitável."""
        if v < 0.5 or v > 100:
            raise ValueError(f'Preço {v} fora do range esperado (0.5-100)')
        return round(v, 2)
    
    @validator('apelido')
    def apelido_limpo(cls, v):
        """Remove espaços extras do apelido."""
        return v.strip()
    
    class Config:
        validate_assignment = True
        extra = 'allow'  # Permite campos extras da API


class PartidaModel(BaseModel):
    """Modelo de validação para dados de partida."""
    
    partida_id: int = Field(..., gt=0)
    clube_casa_id: int = Field(..., gt=0)
    clube_visitante_id: int = Field(..., gt=0)
    rodada: int = Field(..., ge=1, le=38)
    placar_casa: Optional[int] = Field(None, ge=0)
    placar_visitante: Optional[int] = Field(None, ge=0)
    partida_data: Optional[str] = None
    
    @root_validator
    def clubes_diferentes(cls, values):
        """Valida que os clubes são diferentes."""
        casa = values.get('clube_casa_id')
        visitante = values.get('clube_visitante_id')
        if casa == visitante:
            raise ValueError('Clube casa e visitante devem ser diferentes')
        return values


class MercadoStatusModel(BaseModel):
    """Modelo de validação para status do mercado."""
    
    rodada_atual: int = Field(..., ge=1, le=38)
    status_mercado: int = Field(..., ge=1, le=7, description="Status (1=aberto, 2=fechado, etc)")
    fechamento: Optional[str] = Field(None, description="Data/hora de fechamento")
    abertura: Optional[str] = Field(None, description="Data/hora de abertura")
    
    @property
    def mercado_aberto(self) -> bool:
        """Retorna se o mercado está aberto."""
        return self.status_mercado == 1
    
    @property
    def mercado_fechado(self) -> bool:
        """Retorna se o mercado está fechado."""
        return self.status_mercado == 2


class PredicaoModel(BaseModel):
    """Modelo de validação para predições."""
    
    atleta_id: int = Field(..., gt=0)
    predicao: float = Field(..., description="Pontos preditos")
    predicao_std: Optional[float] = Field(None, ge=0, description="Desvio padrão da predição")
    confianca: Optional[float] = Field(None, ge=0, le=1, description="Nível de confiança (0-1)")
    
    @validator('predicao')
    def predicao_razoavel(cls, v):
        """Valida se a predição está em range razoável."""
        if v < -10 or v > 50:
            raise ValueError(f'Predição {v} fora do range esperado (-10 a 50 pontos)')
        return round(v, 2)


class EscalacaoModel(BaseModel):
    """Modelo de validação para escalação completa."""
    
    atletas: List[int] = Field(..., min_items=12, max_items=12, description="IDs dos 12 atletas")
    formacao: str = Field(..., regex=r'^\d-\d-\d$', description="Formação (ex: 4-3-3)")
    patrimonio_usado: float = Field(..., gt=0)
    patrimonio_disponivel: float = Field(..., gt=0)
    pontos_preditos: float = Field(..., description="Total de pontos esperados")
    
    @validator('atletas')
    def atletas_unicos(cls, v):
        """Valida que não há atletas duplicados."""
        if len(v) != len(set(v)):
            raise ValueError('Escalação contém atletas duplicados')
        return v
    
    @root_validator
    def patrimonio_valido(cls, values):
        """Valida que o patrimônio usado não excede o disponível."""
        usado = values.get('patrimonio_usado')
        disponivel = values.get('patrimonio_disponivel')
        if usado > disponivel:
            raise ValueError(f'Patrimônio usado ({usado}) excede disponível ({disponivel})')
        return values


def validar_atleta(data: Dict[str, Any]) -> AtletaModel:
    """Valida dados de atleta.
    
    Args:
        data: Dicionário com dados do atleta
        
    Returns:
        AtletaModel validado
        
    Raises:
        ValidationError: Se dados inválidos
    """
    return AtletaModel(**data)


def validar_atletas_batch(atletas_list: List[Dict[str, Any]]) -> List[AtletaModel]:
    """Valida lista de atletas, ignorando inválidos.
    
    Args:
        atletas_list: Lista de dicionários com dados de atletas
        
    Returns:
        Lista de AtletaModel validados
    """
    validados = []
    erros = []
    
    for i, atleta in enumerate(atletas_list):
        try:
            validados.append(AtletaModel(**atleta))
        except Exception as e:
            erros.append(f"Atleta {i}: {str(e)}")
    
    if erros:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Atletas inválidos ignorados: {len(erros)}")
        for erro in erros[:5]:  # Mostrar apenas primeiros 5
            logger.debug(erro)
    
    return validados
