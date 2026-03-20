"""
PipelineEscalacao – Wrapper legado (mantido para compatibilidade)

Este módulo agora delega toda a lógica para main.py (pipeline unificado).
Para execução standalone via script, use main.py diretamente.

As classes Rodada, Confronto, Jogador, TipoConfrontoEnum ainda podem
ser importadas aqui para uso em testes ou no app.py (Streamlit).
"""

# Re-exporta tipos para compatibilidade
from src.analise_especialista import (
    AnalisadorEspecialista,
    Rodada,
    Confronto,
    Jogador,
    TipoConfrontoEnum,
)

__all__ = [
    "AnalisadorEspecialista",
    "Rodada",
    "Confronto",
    "Jogador",
    "TipoConfrontoEnum",
]
