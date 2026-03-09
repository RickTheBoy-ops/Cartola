"""
Módulo de Validação de Dados para Cartola FC Optimizer
- Validação de atletas, formação, mercado
- Filtragem de atletas inválidos (lesionados, suspensos)
"""

import logging
from typing import Dict, List, Optional
import pandas as pd

logger = logging.getLogger(__name__)

# Status dos atletas na API do Cartola
STATUS_ATLETA = {
    2: 'Dúvida',
    3: 'Suspenso',
    5: 'Contundido',
    6: 'Nulo',
    7: 'Provável',
}

# Status válidos para escalação
STATUS_VALIDOS = {7}  # Apenas "Provável"

# Posições válidas
POSICOES_VALIDAS = {1, 2, 3, 4, 5, 6}

# Campos obrigatórios para um atleta (na raw API response)
CAMPOS_OBRIGATORIOS_ATLETA = ['atleta_id', 'posicao_id']


def validar_mercado(status_data: Dict) -> Dict:
    """
    Valida status do mercado e retorna informações úteis.
    Retorna dict com 'valido', 'rodada_atual', 'mensagem'.
    """
    rodada = status_data.get('rodada_atual')
    status_mercado = status_data.get('status_mercado')

    if rodada is None:
        return {
            'valido': False,
            'rodada_atual': None,
            'mensagem': 'Rodada atual não encontrada na resposta da API'
        }

    # status_mercado: 1 = aberto, 2 = fechado
    mercado_aberto = status_mercado == 1

    return {
        'valido': True,
        'rodada_atual': rodada,
        'mercado_aberto': mercado_aberto,
        'mensagem': f"Rodada {rodada} - Mercado {'ABERTO' if mercado_aberto else 'FECHADO'}"
    }


def validar_atleta(atleta: Dict) -> bool:
    """Valida se um atleta tem os campos mínimos necessários"""
    for campo in CAMPOS_OBRIGATORIOS_ATLETA:
        if campo not in atleta or atleta[campo] is None:
            return False

    # Posição deve ser válida
    if atleta.get('posicao_id') not in POSICOES_VALIDAS:
        return False

    # Preço deve ser positivo
    if atleta.get('preco', atleta.get('preco_num', 0)) <= 0:
        return False

    return True


def filtrar_atletas_validos(atletas_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra DataFrame de atletas removendo:
    - Atletas sem preço
    - Atletas com posição inválida
    - Atletas com status inativo (lesionados, suspensos, etc.)
    """
    if len(atletas_df) == 0:
        return atletas_df.copy()

    df = atletas_df.copy()
    
    # 1. Normalizar nomes das colunas da API se necessário
    col_mapping = {}
    if 'preco_num' in df.columns and 'preco' not in df.columns:
        col_mapping['preco_num'] = 'preco'
    if 'media_num' in df.columns and 'media' not in df.columns:
        col_mapping['media_num'] = 'media'
    if 'variacao_num' in df.columns and 'variacao' not in df.columns:
        col_mapping['variacao_num'] = 'variacao'
        
    if col_mapping:
        df = df.rename(columns=col_mapping)

    # Abortar se preco ainda no existir
    if 'preco' not in df.columns:
        return df

    tam_original = len(df)

    # Filtrar por preço positivo
    df = df[df['preco'] > 0]

    # Filtrar por posição válida
    df = df[df['posicao_id'].isin(POSICOES_VALIDAS)]

    # Filtrar por status (se existir a coluna)
    if 'status_id' in df.columns:
        antes = len(df)
        df = df[df['status_id'].isin(STATUS_VALIDOS)]
        removidos_status = antes - len(df)
        if removidos_status > 0:
            logger.info(f"Removidos {removidos_status} atletas com status inválido (lesão/suspensão/dúvida)")

    removidos_total = tam_original - len(df)
    if removidos_total > 0:
        logger.info(f"Validação: {removidos_total} atletas removidos, {len(df)} válidos restantes")

    return df.reset_index(drop=True)


def validar_formacao(formacao: str) -> bool:
    """Valida se a formação é reconhecida"""
    formacoes_validas = {'3-4-3', '3-5-2', '4-3-3', '4-4-2', '4-5-1', '5-3-2', '5-4-1'}
    return formacao in formacoes_validas


def validar_historico_minimo(df: pd.DataFrame, minimo: int = 30) -> bool:
    """Verifica se há histórico mínimo suficiente para treinar ML"""
    if df is None or len(df) == 0:
        return False
    return len(df) >= minimo


def validar_time(team: List[Dict], patrimonio: float, formacao_nome: str) -> Dict:
    """
    Valida um time gerado pelo otimizador.
    Retorna dict com 'valido', 'erros'.
    """
    erros = []

    if not team:
        return {'valido': False, 'erros': ['Time vazio']}

    # Verificar preço total
    total_preco = sum(a.get('preco', 0) for a in team)
    if total_preco > patrimonio:
        erros.append(f"Preço total (C$ {total_preco:.2f}) excede patrimônio (C$ {patrimonio:.2f})")

    # Verificar duplicatas
    ids = [a.get('atleta_id') for a in team]
    if len(ids) != len(set(ids)):
        erros.append("Time contém jogadores duplicados")

    # Quantidade total de jogadores (deve ser 12)
    if len(team) != 12:
        erros.append(f"Time com {len(team)} jogadores (esperado: 12)")

    return {
        'valido': len(erros) == 0,
        'erros': erros,
        'total_preco': total_preco,
        'total_jogadores': len(team)
    }


def validar_partida_confirmada(clube_id: int, rodada: int, partidas_df) -> bool:
    """
    Verifica se um clube tem partida confirmada em uma determinada rodada.
    Retorna True se existe jogo, False caso contrário.

    Args:
        clube_id: ID do clube do atleta
        rodada: Número da rodada a verificar
        partidas_df: DataFrame com colunas [rodada, clube_casa_id, clube_visitante_id]
    """
    if partidas_df is None or len(partidas_df) == 0:
        logger.debug(f"Sem dados de partidas para validar clube {clube_id} na rodada {rodada}")
        return True  # Sem dados → assumir que joga (não bloquear)

    colunas_necessarias = {'rodada', 'clube_casa_id', 'clube_visitante_id'}
    if not colunas_necessarias.issubset(partidas_df.columns):
        return True  # Estrutura incompatível → não bloquear

    partida = partidas_df[
        (partidas_df['rodada'] == rodada) &
        (
            (partidas_df['clube_casa_id'] == clube_id) |
            (partidas_df['clube_visitante_id'] == clube_id)
        )
    ]

    if len(partida) == 0:
        logger.warning(
            f"⚠️ Clube {clube_id} não tem partida confirmada na rodada {rodada}. "
            f"Atletas deste clube serão ignorados."
        )
        return False

    return True


def filtrar_atletas_com_jogo(
    atletas_df,
    rodada: int,
    partidas_df,
) -> 'pd.DataFrame':
    """
    Remove atletas cujo clube não tem partida confirmada na rodada especificada.
    Retorna DataFrame filtrado.
    """
    import pandas as pd

    if 'clube_id' not in atletas_df.columns:
        return atletas_df

    if partidas_df is None or len(partidas_df) == 0:
        return atletas_df

    clubes_com_jogo = set()

    # Mandantes
    if 'clube_casa_id' in partidas_df.columns:
        rodada_partidas = partidas_df[partidas_df['rodada'] == rodada]
        clubes_com_jogo.update(rodada_partidas['clube_casa_id'].dropna().astype(int))
        clubes_com_jogo.update(rodada_partidas['clube_visitante_id'].dropna().astype(int))

    antes = len(atletas_df)
    filtrado = atletas_df[atletas_df['clube_id'].isin(clubes_com_jogo)].copy()
    removidos = antes - len(filtrado)

    if removidos > 0:
        logger.warning(
            f"⚠️ {removidos} atletas removidos por clube sem partida na rodada {rodada}. "
            f"{len(filtrado)} atletas mantidos."
        )

    return filtrado.reset_index(drop=True)
