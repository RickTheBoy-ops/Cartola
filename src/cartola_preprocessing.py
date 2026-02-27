#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
========================================================================
CARTOLA FC - PREPROCESSING UTILITIES
========================================================================
Funções de limpeza e transformação de dados
Extraídas do projeto caRtola-master (sem dependência do Kedro)

Funcionalidades:
- Limpeza de dados
- Padronização de colunas
- Preenchimento de scouts
- Remoção de duplicatas
- Concatenação de datasets
========================================================================
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path


# ========================================================================
# FUNÇÕES BÁSICAS DE LIMPEZA
# ========================================================================

def drop_duplicated_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove linhas duplicadas do DataFrame
    
    Args:
        df: DataFrame de entrada
        
    Returns:
        DataFrame sem duplicatas
    """
    return df.drop_duplicates(ignore_index=True)


def rename_cols(df: pd.DataFrame, map_col_names: Dict[str, str]) -> pd.DataFrame:
    """
    Renomeia colunas do DataFrame
    
    Args:
        df: DataFrame de entrada
        map_col_names: Dicionário {nome_antigo: nome_novo}
        
    Returns:
        DataFrame com colunas renomeadas
    """
    return df.rename(columns=map_col_names)


def concat_partitioned_datasets(dataframes: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatena múltiplos DataFrames
    
    Args:
        dataframes: Lista de DataFrames para concatenar
        
    Returns:
        DataFrame concatenado
    """
    
    if not dataframes:
        return pd.DataFrame()
    
    df_concat = pd.concat(dataframes, ignore_index=True)
    return df_concat.reset_index(drop=True)


# ========================================================================
# FUNÇÕES ESPECÍFICAS DO CARTOLA
# ========================================================================

def map_status_id_to_string(df: pd.DataFrame, mapping: Dict[int, str] = None) -> pd.DataFrame:
    """
    Converte status_id numérico para string legível
    
    Args:
        df: DataFrame com coluna 'status_id' ou 'atletas.status_id'
        mapping: Dicionário opcional {id: string}
        
    Returns:
        DataFrame com nova coluna 'status_str'
    """
    
    # Mapeamento padrão se não fornecido
    if mapping is None:
        mapping = {
            2: "Dúvida",
            3: "Suspenso",
            5: "Contundido",
            6: "Nulo",
            7: "Provável"
        }
    
    # Detectar nome da coluna
    if 'status_id' in df.columns:
        col_name = 'status_id'
    elif 'atletas.status_id' in df.columns:
        col_name = 'atletas.status_id'
    else:
        return df
    
    # Mapear
    df['status_str'] = df[col_name].map(mapping).fillna('Desconhecido')
    
    return df


def fill_scouts_with_zeros(df: pd.DataFrame, scout_cols: List[str] = None) -> pd.DataFrame:
    """
    Preenche colunas de scout vazias com zeros
    
    Args:
        df: DataFrame
        scout_cols: Lista de colunas de scout. Se None, detecta automaticamente (UPPER)
        
    Returns:
        DataFrame com scouts preenchidos
    """
    
    # Detectar colunas de scout automaticamente (maiúsculas)
    if scout_cols is None:
        scout_cols = [col for col in df.columns if col.isupper() and len(col) <= 4]
    
    # Preencher com zeros
    for col in scout_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    return df


def fill_empty_slugs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preenche slugs vazios com versão simplificada do apelido
    
    Args:
        df: DataFrame com colunas 'slug' e 'apelido' (ou atletas.*)
        
    Returns:
        DataFrame com slugs preenchidos
    """
    
    # Detectar nomes das colunas
    slug_col = 'slug' if 'slug' in df.columns else 'atletas.slug'
    apelido_col = 'apelido' if 'apelido' in df.columns else 'atletas.apelido'
    
    if slug_col not in df.columns or apelido_col not in df.columns:
        return df
    
    # Preencher slugs vazios
    def create_slug(row):
        if pd.isna(row[slug_col]) or row[slug_col] == '':
            # Criar slug simples do apelido (lowercase, sem acentos)
            from unidecode import unidecode
            apelido = str(row[apelido_col])
            slug = unidecode(apelido).lower().replace(' ', '-')
            return slug
        return row[slug_col]
    
    df[slug_col] = df.apply(create_slug, axis=1)
    
    return df


# ========================================================================
# FUNÇÕES DE CARGA DE DADOS
# ========================================================================

def load_round_data(year: int, round_num: int, data_dir: str = "data") -> Optional[pd.DataFrame]:
    """
    Carrega dados de uma rodada específica
    
    Args:
        year: Ano (ex: 2024)
        round_num: Número da rodada (ex: 15)
        data_dir: Diretório base de dados
        
    Returns:
        DataFrame ou None se não encontrado
    """
    
    filepath = Path(data_dir) / "historical" / str(year) / f"rodada-{round_num}.csv"
    
    if not filepath.exists():
        print(f"⚠️ Arquivo não encontrado: {filepath}")
        return None
    
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Carregado: {len(df)} linhas de {filepath.name}")
        return df
    except Exception as e:
        print(f"❌ Erro ao carregar {filepath}: {e}")
        return None


def load_multiple_rounds(year_start: int, year_end: int = None, 
                        data_dir: str = "data") -> pd.DataFrame:
    """
    Carrega e concatena dados de múltiplas rodadas
    
    Args:
        year_start: Ano inicial
        year_end: Ano final (se None, usa apenas year_start)
        data_dir: Diretório base
        
    Returns:
        DataFrame concatenado
    """
    
    if year_end is None:
        year_end = year_start
    
    dataframes = []
    
    for year in range(year_start, year_end + 1):
        year_dir = Path(data_dir) / "historical" / str(year)
        
        if not year_dir.exists():
            print(f"⚠️ Diretório não encontrado: {year_dir}")
            continue
        
        # Listar todos os CSVs da rodada
        csv_files = sorted(year_dir.glob("rodada-*.csv"))
        
        print(f"\n📂 Ano {year}: {len(csv_files)} rodadas encontradas")
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                df['year'] = year
                df['round'] = int(csv_file.stem.replace('rodada-', ''))
                dataframes.append(df)
            except Exception as e:
                print(f"   ⚠️ Erro em {csv_file.name}: {e}")
    
    if not dataframes:
        print("❌ Nenhum dado carregado!")
        return pd.DataFrame()
    
    df_concat = concat_partitioned_datasets(dataframes)
    
    print(f"\n✅ Total: {len(df_concat)} linhas de {len(dataframes)} rodadas")
    
    return df_concat


# ========================================================================
# PIPELINE COMPLETO
# ========================================================================

def preprocess_cartola_data(df: pd.DataFrame, 
                           fill_scouts: bool = True,
                           fill_slugs: bool = True,
                           remove_duplicates: bool = True) -> pd.DataFrame:
    """
    Pipeline completo de preprocessamento
    
    Args:
        df: DataFrame bruto
        fill_scouts: Se True, preenche scouts com zeros
        fill_slugs: Se True, preenche slugs vazios
        remove_duplicates: Se True, remove duplicatas
        
    Returns:
        DataFrame limpo
    """
    
    print("🔄 Iniciando preprocessamento...")
    
    # Status ID para string
    df = map_status_id_to_string(df)
    
    # Preencher scouts
    if fill_scouts:
        df = fill_scouts_with_zeros(df)
        print("   ✅ Scouts preenchidos")
    
    # Preencher slugs
    if fill_slugs:
        try:
            df = fill_empty_slugs(df)
            print("   ✅ Slugs preenchidos")
        except ImportError:
            print("   ⚠️ unidecode não instalado, pulando fill_slugs")
    
    # Remover duplicatas
    if remove_duplicates:
        len_before = len(df)
        df = drop_duplicated_rows(df)
        removed = len_before - len(df)
        if removed > 0:
            print(f"   ✅ {removed} duplicatas removidas")
    
    print("✅ Preprocessamento concluído!")
    
    return df


# ========================================================================
# MAPEAMENTOS PADRÃO
# ========================================================================

# Mapeamento de colunas (caso precise renomear)
COLUMN_MAPPING = {
    'atleta_id': 'id',
    'apelido': 'nickname',
    'clube_id': 'team_id',
    'posicao_id': 'position_id',
    'status_id': 'status_id',
    'pontos_num': 'points',
    'preco_num': 'price',
    'media_num': 'avg',
    'variacao_num': 'variation'
}


# Scout cols padrão
SCOUT_COLS = [
    'A', 'CA', 'CV', 'DD', 'DP', 'FC', 'FD', 'FF', 'FS', 'FT',
    'G', 'GC', 'GS', 'I', 'PE', 'PI', 'PP', 'PS', 'RB', 'SG', 'V'
]


# ========================================================================
# CLI / TESTES
# ========================================================================

def main():
    """Teste básico das funções"""
    
    print("\n" + "="*80)
    print("🧹 CARTOLA FC - PREPROCESSING UTILITIES")
    print("="*80 + "\n")
    
    # Teste: carregar rodada
    print("Teste 1: Carregar rodada")
    df = load_round_data(2024, 1)
    
    if df is not None:
        print(f"\nColunas: {list(df.columns[:10])}...")
        print(f"Shape: {df.shape}")
        
        # Teste: preprocessar
        print("\nTeste 2: Preprocessamento")
        df_clean = preprocess_cartola_data(df)
        
        print(f"\nShape após limpeza: {df_clean.shape}")
    
    # Teste: carregar múltiplas rodadas
    print("\n" + "-"*80)
    print("Teste 3: Carregar múltiplas rodadas")
    df_multi = load_multiple_rounds(2024, 2024)
    
    if len(df_multi) > 0:
        print(f"\nTotal de linhas: {len(df_multi)}")
        print(f"Anos únicos: {df_multi['year'].unique()}")
        print(f"Rodadas únicas: {sorted(df_multi['round'].unique())}")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
