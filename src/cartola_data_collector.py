#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
========================================================================
CARTOLA FC - DATA COLLECTOR
========================================================================
Download automático de dados da API do Cartola FC
Baseado no projeto caRtola-master (github.com/henriquepgomide/caRtola)

Funcionalidades:
- Download da rodada atual
- Salvamento automático em CSV
- Organização por ano/rodada
- Detecção de nova rodada
========================================================================
"""

import os
from datetime import date, datetime
from pathlib import Path
import pandas as pd
import requests
import json
from typing import Optional, Dict

from src.utils.exceptions import APIConnectionError, DataValidationError


# ========================================================================
# CONFIGURAÇÃO
# ========================================================================

API_ATLETAS = "https://api.cartola.globo.com/atletas/mercado"
API_CLUBES = "https://api.cartola.globo.com/clubes"
API_PARTIDAS = "https://api.cartola.globo.com/partidas"

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

# Diretórios
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CURRENT_DIR = DATA_DIR / "current"
HISTORICAL_DIR = DATA_DIR / "historical"


# ========================================================================
# FUNÇÕES DE DOWNLOAD
# ========================================================================

def download_current_data() -> Optional[Dict]:
    """
    Baixa dados atuais da API do Cartola FC
    
    Returns:
        Dict com dados completos ou None se erro
    """
    
    print("\n📡 Baixando dados do Cartola FC...")
    
    try:
        # Atletas
        r_atletas = requests.get(API_ATLETAS, headers=HEADERS, timeout=30)
        r_atletas.raise_for_status()
        data_atletas = r_atletas.json()
        
        # Clubes
        r_clubes = requests.get(API_CLUBES, headers=HEADERS, timeout=30)
        r_clubes.raise_for_status()
        data_clubes = r_clubes.json()
        
        # Partidas
        try:
            r_partidas = requests.get(API_PARTIDAS, headers=HEADERS, timeout=30)
            r_partidas.raise_for_status()
            data_partidas = r_partidas.json()
        except:
            data_partidas = {}
            print("   ⚠️ Partidas não disponíveis")
        
        print("   ✅ Dados baixados com sucesso!")
        
        return {
            'atletas': data_atletas,
            'clubes': data_clubes,
            'partidas': data_partidas
        }
    
    except requests.RequestException as e:
        print(f"   ❌ Erro de conexão com a API: {e}")
        raise APIConnectionError(f"Falha na API: {e}")
    except Exception as e:
        print(f"   ❌ Erro inesperado ao baixar dados: {e}")
        raise APIConnectionError(f"Erro inesperado: {e}")


def process_data_to_dataframe(data: Dict) -> Optional[pd.DataFrame]:
    """
    Processa dados brutos da API em DataFrame
    Adaptado do caRtola-master/download_data.py
    
    Args:
        data: Dict com atletas, clubes e partidas
        
    Returns:
        DataFrame processado ou None se erro
    """
    
    print("🔄 Processando dados...")
    
    try:
        # Atletas
        atletas_dict = data['atletas']['atletas']
        
        # Converter para DataFrame
        if isinstance(atletas_dict, dict):
            df_atletas = pd.DataFrame(atletas_dict.values())
        else:
            df_atletas = pd.DataFrame(atletas_dict)
        
        # Expandir scout (se existir)
        if 'scout' in df_atletas.columns:
            scout_data = df_atletas.pop('scout')
            if not scout_data.isna().all():
                df_scout = pd.DataFrame(scout_data.values.tolist())
                df_atletas = df_atletas.join(df_scout)
        
        # Renomear colunas de atletas
        df_atletas = df_atletas.rename(
            columns={
                col: f"atletas.{col}" if col.islower() else col 
                for col in df_atletas.columns
            }
        )
        
        # Clubes
        df_clubes = pd.DataFrame(data['clubes'].values())
        df_clubes = df_clubes.rename(
            columns={
                'id': 'atletas.clube_id',
                'nome': 'atletas.clube.id.full.name'
            }
        )
        df_clubes = df_clubes[['atletas.clube_id', 'atletas.clube.id.full.name']]
        
        # Merge
        df_merge = df_atletas.merge(df_clubes, how='left', on='atletas.clube_id')
        
        # Organizar colunas (scouts no final)
        cols_scouts = [col for col in df_merge.columns if col.isupper()]
        cols_atleta = list(set(df_merge.columns) - set(cols_scouts))
        
        df_merge = df_merge.loc[:, cols_atleta + cols_scouts]
        df_merge.sort_values(by='atletas.atleta_id', inplace=True, ignore_index=True)
        
        print(f"   ✅ {len(df_merge)} atletas processados")
        
        return df_merge
    
    except KeyError as e:
        print(f"   ❌ Chave ausente nos dados: {e}")
        raise DataValidationError(f"Dados mal estruturados da API. Faltando chave: {e}")
    except Exception as e:
        print(f"   ❌ Erro ao processar dados esportivos: {e}")
        raise DataValidationError(f"Erro no processamento esportivo: {e}")


def get_round_info(df: pd.DataFrame) -> tuple:
    """
    Extrai informações da rodada do DataFrame
    
    Returns:
        Tuple (rodada_num, ano)
    """
    
    if 'atletas.rodada_id' in df.columns:
        rodada = int(df.loc[0, 'atletas.rodada_id'])
    else:
        rodada = 1
    
    ano = date.today().year
    
    return rodada, ano


def save_to_csv(df: pd.DataFrame, rodada: int, ano: int, historical: bool = False) -> str:
    """
    Salva DataFrame em CSV
    
    Args:
        df: DataFrame para salvar
        rodada: Número da rodada
        ano: Ano
        historical: Se True, salva em historical/, senão em current/
        
    Returns:
        Path do arquivo salvo
    """
    
    # Diretório base
    if historical:
        base_dir = HISTORICAL_DIR / str(ano)
    else:
        base_dir = CURRENT_DIR
    
    # Criar diretório se não existir
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Arquivo
    filename = f"rodada-{rodada}.csv"
    filepath = base_dir / filename
    
    # Salvar
    df.to_csv(filepath, index=False)
    
    print(f"💾 Salvo em: {filepath}")
    
    return str(filepath)


def save_json(data: Dict, rodada: int, ano: int) -> str:
    """
    Salva dados brutos em JSON (backup)
    
    Returns:
        Path do arquivo salvo
    """
    
    base_dir = CURRENT_DIR
    base_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"rodada-{rodada}.json"
    filepath = base_dir / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 JSON salvo em: {filepath}")
    
    return str(filepath)


# ========================================================================
# FUNÇÕES PRINCIPAIS
# ========================================================================

def download_and_save_current_round(save_json_backup: bool = True) -> Optional[str]:
    """
    Baixa e salva a rodada atual
    
    Args:
        save_json_backup: Se True, salva também JSON com dados brutos
        
    Returns:
        Path do CSV salvo ou None se erro
    """
    
    print("\n" + "="*80)
    print("📥 DOWNLOAD DA RODADA ATUAL")
    print("="*80)
    
    # Download
    data = None
    try:
        data = download_current_data()
    except APIConnectionError as e:
        print(f"⚠️ Operação abortada devido a erro de conexão: {e}")
        return None
        
    if not data:
        return None
    
    # Processar
    df = None
    try:
        df = process_data_to_dataframe(data)
    except DataValidationError as e:
        print(f"⚠️ Operação abortada devido a erro de validação: {e}")
        return None
        
    if df is None:
        return None
    
    # Info da rodada
    rodada, ano = get_round_info(df)
    print(f"\n🎯 Rodada: {rodada} | Ano: {ano}")
    
    # Salvar CSV
    csv_path = save_to_csv(df, rodada, ano, historical=False)
    
    # Salvar JSON (opcional)
    if save_json_backup:
        save_json(data, rodada, ano)
    
    print("\n" + "="*80)
    print("✅ DOWNLOAD CONCLUÍDO!")
    print("="*80 + "\n")
    
    return csv_path


def get_latest_round() -> tuple:
    """
    Retorna informações da última rodada baixada
    
    Returns:
        Tuple (rodada, ano, filepath) ou (None, None, None)
    """
    
    if not CURRENT_DIR.exists():
        return None, None, None
    
    # Procurar arquivos CSV
    csv_files = list(CURRENT_DIR.glob("rodada-*.csv"))
    
    if not csv_files:
        return None, None, None
    
    # Último arquivo (mais recente)
    latest_file = max(csv_files, key=lambda p: p.stat().st_mtime)
    
    # Extrair rodada do nome do arquivo
    rodada_str = latest_file.stem.replace("rodada-", "")
    try:
        rodada = int(rodada_str)
    except:
        rodada = None
    
    ano = date.today().year
    
    return rodada, ano, str(latest_file)


def check_new_round() -> bool:
    """
    Verifica se há uma nova rodada disponível
    
    Returns:
        True se há nova rodada, False caso contrário
    """
    
    print("🔍 Verificando nova rodada...")
    
    try:
        # Baixar info atual
        r = requests.get(API_ATLETAS, headers=HEADERS, timeout=10)
        data = r.json()
        
        atletas = data.get('atletas', {})
        if isinstance(atletas, dict):
            atletas = list(atletas.values())
        
        if not atletas:
            print("   ⚠️ Nenhum atleta encontrado")
            return False
        
        rodada_api = atletas[0].get('rodada_id')
        
        # Última rodada salva
        rodada_local, _, _ = get_latest_round()
        
        if rodada_local is None:
            print(f"   ✅ Nova rodada {rodada_api} disponível (primeira vez)")
            return True
        
        if rodada_api > rodada_local:
            print(f"   ✅ Nova rodada {rodada_api} disponível (anterior: {rodada_local})")
            return True
        
        print(f"   ℹ️ Rodada {rodada_api} já baixada")
        return False
    
    except Exception as e:
        print(f"   ❌ Erro ao verificar: {e}")
        return False


def copy_current_to_historical():
    """
    Copia rodada de current/ para historical/
    """
    
    rodada, ano, filepath = get_latest_round()
    
    if not filepath:
        print("⚠️ Nenhuma rodada em current/ para copiar")
        return
    
    df = pd.read_csv(filepath)
    save_to_csv(df, rodada, ano, historical=True)
    print(f"📋 Copiado para historical/{ano}/rodada-{rodada}.csv")


# ========================================================================
# CLI
# ========================================================================

def main():
    """Interface de linha de comando"""
    
    print("\n" + "="*80)
    print("📥 CARTOLA FC - DATA COLLECTOR")
    print("="*80)
    print("\n1 - Download da rodada atual")
    print("2 - Verificar nova rodada")
    print("3 - Ver última rodada baixada")
    print("4 - Copiar current → historical")
    print("0 - Sair")
    print("\n" + "="*80)
    
    escolha = input("\nEscolha: ").strip()
    
    if escolha == '1':
        download_and_save_current_round()
    
    elif escolha == '2':
        check_new_round()
    
    elif escolha == '3':
        rodada, ano, filepath = get_latest_round()
        if rodada:
            print(f"\n📊 Última rodada: {rodada} ({ano})")
            print(f"📁 Arquivo: {filepath}")
        else:
            print("\n⚠️ Nenhuma rodada baixada ainda")
    
    elif escolha == '4':
        copy_current_to_historical()
    
    elif escolha == '0':
        print("\n👋 Até logo!\n")
    
    else:
        print("\n⚠️ Opção inválida!\n")


if __name__ == "__main__":
    main()
