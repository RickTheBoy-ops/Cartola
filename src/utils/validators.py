"""
Validadores do sistema Cartola FC.

Responsabilidade: validar dados brutos antes que entrem no pipeline.
Não deve conter lógica de negócio (cálculo de xpoints, otimização, etc.).
"""

import logging
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Status que permitem escalação
STATUS_PROVAVEL = 7
STATUS_DUVIDA   = [2, 3, 5]   # Dúvida, Contundido, Suspenso

# Formações válidas
FORMACOES_VALIDAS = {'3-4-3','3-5-2','4-3-3','4-4-2','4-5-1','5-3-2','5-4-1'}


# ==============================================================
# ATLETA
# ==============================================================

def validar_atleta(atleta: Dict) -> bool:
    """
    Valida se o atleta tem os campos obrigatórios mínimos.
    NÃO filtra por status aqui — isso é feito em filtrar_atletas_validos.
    """
    if not isinstance(atleta, dict):
        return False
    return bool(atleta.get('atleta_id')) and bool(atleta.get('posicao_id'))


def filtrar_atletas_validos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove atletas inválidos para escalação:
      - Sem atleta_id ou posicao_id
      - status_id != STATUS_PROVAVEL (7)
      - Preço zero ou negativo

    Esta é a trava estrutural: jogadores duvidosos/suspensos/contundidos
    nunca chegam ao otimizador.
    """
    if df.empty:
        return df

    inicial = len(df)

    # Campos obrigatórios
    df = df.dropna(subset=['atleta_id', 'posicao_id'])

    # Trava de status: só status_id == 7 (Provável)
    if 'status_id' in df.columns:
        antes_status = len(df)
        df = df[df['status_id'] == STATUS_PROVAVEL]
        removidos_status = antes_status - len(df)
        if removidos_status > 0:
            logger.info(
                f"🧹 {removidos_status} atletas removidos (status != {STATUS_PROVAVEL}: "
                f"dúvida/contundido/suspenso)"
            )

    # Preço mínimo
    if 'preco' in df.columns:
        df = df[df['preco'] > 0]

    total_removidos = inicial - len(df)
    logger.info(
        f"✅ filtrar_atletas_validos: {len(df)} válidos | "
        f"{total_removidos} removidos do total de {inicial}"
    )
    return df.reset_index(drop=True)


# ==============================================================
# MERCADO
# ==============================================================

def validar_mercado(mercado: Dict) -> Dict:
    """
    Valida status do mercado.
    Retorna dict com 'valido', 'rodada_atual' e 'mensagem'.
    """
    if not mercado:
        return {'valido': False, 'rodada_atual': 0, 'mensagem': 'Mercado não retornou dados'}

    rodada = mercado.get('rodada_atual', 0)
    status = mercado.get('status_mercado', 0)
    nome   = mercado.get('nome_status', 'Desconhecido')

    # Status 1=Fechado, 2=Atualizando, 4=Manutenção, 6=Aberto, 7=Parcial...
    if status in [4]:
        return {
            'valido': False,
            'rodada_atual': rodada,
            'mensagem': f'Mercado em manutenção (status={status}: {nome})'
        }

    return {
        'valido':       True,
        'rodada_atual': int(rodada),
        'mensagem':     f'Rodada {rodada} | Status: {nome} (id={status})'
    }


# ==============================================================
# FORMAÇÃO
# ==============================================================

def validar_formacao(formacao: str) -> bool:
    """Valida se a formação é suportada pelo sistema."""
    ok = formacao in FORMACOES_VALIDAS
    if not ok:
        logger.warning(
            f"❌ Formação '{formacao}' inválida. Válidas: {sorted(FORMACOES_VALIDAS)}"
        )
    return ok


# ==============================================================
# TIME FINALIZADO
# ==============================================================

def validar_time(
    team: list,
    patrimonio: float,
    formacao: str,
    max_clube: int = 3
) -> Dict:
    """
    Valida o time final antes de exibir a escalação.
    Retorna {'valido': bool, 'erros': list[str], 'alertas': list[str]}
    """
    erros   = []
    alertas = []

    if not team:
        return {'valido': False, 'erros': ['Time vazio'], 'alertas': []}

    total_preco = sum(a.get('preco', 0) for a in team)
    if total_preco > patrimonio:
        erros.append(f'Custo C${total_preco:.2f} excede orçamento C${patrimonio:.2f}')

    # Verificar duplicatas
    ids = [a.get('atleta_id') for a in team]
    if len(ids) != len(set(ids)):
        erros.append('Time com atletas duplicados!')

    # Verificar status de todos (nenhum deve ter passado com status != 7)
    invalidos = [a.get('apelido','?') for a in team if int(a.get('status_id', 7)) != STATUS_PROVAVEL]
    if invalidos:
        erros.append(f'Atletas não prováveis no time: {invalidos}')

    # Concentração por clube
    from collections import Counter
    por_clube = Counter(a.get('clube_id') for a in team)
    for clube, cnt in por_clube.items():
        if cnt > max_clube:
            alertas.append(f'Clube {clube}: {cnt} atletas (máx recomendado: {max_clube})')

    return {
        'valido':   len(erros) == 0,
        'erros':    erros,
        'alertas':  alertas,
    }


# ==============================================================
# FILTROS AUXILIARES
# ==============================================================

def filtrar_atletas_com_jogo(
    atletas_df: pd.DataFrame,
    rodada: int,
    partidas_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Remove atletas cujo clube não tem jogo confirmado na rodada.
    """
    if partidas_df.empty:
        return atletas_df

    partidas_rodada = partidas_df[partidas_df['rodada'] == rodada]
    clubes_com_jogo = set()

    for col in ['clube_casa_id', 'clube_visitante_id', 'clube_id_a', 'clube_id_b']:
        if col in partidas_rodada.columns:
            clubes_com_jogo.update(partidas_rodada[col].dropna().astype(int).tolist())

    if not clubes_com_jogo:
        logger.warning("⚠️  Nenhum clube com jogo confirmado. Retornando todos os atletas.")
        return atletas_df

    filtrado = atletas_df[atletas_df['clube_id'].isin(clubes_com_jogo)]
    removidos = len(atletas_df) - len(filtrado)
    if removidos:
        logger.info(f"🔍 {removidos} atletas sem jogo confirmado removidos")
    return filtrado.reset_index(drop=True)
