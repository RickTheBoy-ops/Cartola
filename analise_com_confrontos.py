#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
========================================================================
CARTOLA FC - ANÁLISE RÁPIDA COM CONFRONTOS
========================================================================
Análise que considera os confrontos da rodada para recomendações
========================================================================
"""

import requests
from datetime import datetime

def carregar_dados():
    """Carrega dados da API"""
    
    API_ATLETAS = "https://api.cartola.globo.com/atletas/mercado"
    API_CLUBES = "https://api.cartola.globo.com/clubes"
    API_PARTIDAS = "https://api.cartola.globo.com/partidas"
    
    HEADERS = {'User-Agent': 'Mozilla/5.0'}
    
    print("\n" + "="*80)
    print("ANALISE COM CONFRONTOS - CARTOLA FC")
    print("="*80)
    print(f"\nHorario: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
    
    try:
        r1 = requests.get(API_ATLETAS, headers=HEADERS, timeout=30)
        r2 = requests.get(API_CLUBES, headers=HEADERS, timeout=30)
        r3 = requests.get(API_PARTIDAS, headers=HEADERS, timeout=30)
        
        return r1.json(), r2.json(), r3.json()
    except Exception as e:
        print(f"ERRO: {e}")
        return None, None, None


def processar_confrontos(dados_partidas, dados_clubes):
    """Processa confrontos da rodada"""
    
    clubes_dict = {int(k): v.get('nome', '?') for k, v in dados_clubes.items()}
    
    # Extrair partidas
    partidas_lista = []
    if isinstance(dados_partidas, dict):
        for key, value in dados_partidas.items():
            if isinstance(value, dict) and 'partidas' in value:
                partidas_lista.extend(value['partidas'])
            elif isinstance(value, list):
                partidas_lista.extend(value)
            elif isinstance(value, dict) and 'clube_casa_id' in value:
                partidas_lista.append(value)
    
    # Mapear cada clube -> adversário
    confrontos = {}
    
    print("="*80)
    print("CONFRONTOS DA RODADA")
    print("="*80)
    print()
    
    for partida in partidas_lista:
        if not isinstance(partida, dict):
            continue
            
        casa_id = partida.get('clube_casa_id')
        visitante_id = partida.get('clube_visitante_id')
        
        if casa_id and visitante_id:
            casa_nome = clubes_dict.get(casa_id, 'Desconhecido')
            visitante_nome = clubes_dict.get(visitante_id, 'Desconhecido')
            
            print(f"   {casa_nome:25} (casa)  vs  {visitante_nome:25} (fora)")
            
            # Mapear confronto
            confrontos[casa_id] = {
                'adversario_id': visitante_id,
                'adversario': visitante_nome,
                'mando': 'casa'
            }
            confrontos[visitante_id] = {
                'adversario_id': casa_id,
                'adversario': casa_nome,
                'mando': 'fora'
            }
    
    print()
    return confrontos


def analisar_com_confrontos(dados_atletas, dados_clubes, confrontos):
    """Analisa jogadores considerando confrontos"""
    
    clubes_dict = {int(k): v.get('nome', '?') for k, v in dados_clubes.items()}
    
    atletas = dados_atletas.get('atletas', [])
    if isinstance(atletas, dict):
        atletas = list(atletas.values())
    
    # Filtrar prováveis
    atletas_provaveis = [a for a in atletas if a.get('status_id') == 7]
    
    POS_MAP = {1: "Goleiro", 2: "Lateral", 3: "Zagueiro", 4: "Meia", 5: "Atacante", 6: "Técnico"}
    
    print("="*80)
    print("TOP JOGADORES POR POSICAO (com adversarios)")
    print("="*80)
    print()
    
    recomendacoes = []
    
    for pos_id in [1, 2, 3, 4, 5]:
        pos_nome = POS_MAP[pos_id]
        
        # Filtrar e ordenar
        pos_atletas = [a for a in atletas_provaveis if a.get('posicao_id') == pos_id]
        pos_atletas.sort(key=lambda x: x.get('media_num', 0), reverse=True)
        
        print(f"\n{pos_nome.upper()}:")
        print("-"*80)
        print(f"{'Jogador':20} {'Time':15} {'Media':>6} {'Preco':>8} {'Adversario':25} {'Mando':6}")
        print("-"*80)
        
        for i, atleta in enumerate(pos_atletas[:10], 1):
            nome = atleta.get('apelido', '?')[:19]
            clube_id = atleta.get('clube_id')
            clube = clubes_dict.get(clube_id, '?')[:14]
            media = atleta.get('media_num', 0)
            preco = atleta.get('preco_num', 0)
            
            # Buscar adversário
            confronto = confrontos.get(clube_id, {})
            adversario = confronto.get('adversario', 'Sem jogo')[:24]
            mando = confronto.get('mando', '-')[:5]
            
            print(f"{i:2}. {nome:18} {clube:15} {media:6.2f} C${preco:6.2f} vs {adversario:23} ({mando})")
            
            # Adicionar às recomendações se média boa e custo-benefício
            if i <= 5 and media >= 8.0:
                custo_beneficio = media / max(preco, 0.1)
                recomendacoes.append({
                    'nome': nome,
                    'posicao': pos_nome,
                    'clube': clube,
                    'media': media,
                    'preco': preco,
                    'adversario': adversario,
                    'mando': mando,
                    'custo_beneficio': custo_beneficio
                })
    
    print()
    return recomendacoes


def gerar_recomendacoes(recomendacoes):
    """Gera recomendações finais"""
    
    print("\n" + "="*80)
    print("RECOMENDACOES BASEADAS EM CONFRONTOS")
    print("="*80)
    print()
    
    # Ordenar por custo-benefício
    recomendacoes.sort(key=lambda x: x['custo_beneficio'], reverse=True)
    
    print("TOP 15 JOGADORES (Custo-Beneficio + Confronto):")
    print("-"*80)
    print(f"{'Jogador':20} {'Pos':10} {'Time':15} {'Media':>6} {'Preco':>8} vs {'Adversario':20}")
    print("-"*80)
    
    for i, rec in enumerate(recomendacoes[:15], 1):
        print(f"{i:2}. {rec['nome']:18} {rec['posicao']:10} {rec['clube']:15} "
              f"{rec['media']:6.2f} C${rec['preco']:6.2f} vs {rec['adversario']:18} ({rec['mando']})")
    
    print()
    print("="*80)
    print("DICAS ESTRATEGICAS:")
    print("="*80)
    print()
    print("1. Times JOGANDO EM CASA tendem a pontuar mais (defensores + atacantes)")
    print("2. Observe confrontos faceis (time forte vs fraco)")
    print("3. Evite jogadores contra defesas solidas")
    print("4. Prefira atacantes/meias em jogos com muitos gols esperados")
    print("5. Goleiros/defensores em confrontos equilibrados ou favoraveis")
    print()
    print("="*80)
    
    # Salvar relatório
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"analise_confrontos_{timestamp}.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("ANALISE COM CONFRONTOS - CARTOLA FC\n")
        f.write("="*80 + "\n\n")
        f.write(f"Data/Hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n")
        
        f.write("TOP 15 RECOMENDACOES:\n")
        f.write("-"*80 + "\n\n")
        
        for i, rec in enumerate(recomendacoes[:15], 1):
            f.write(f"{i}. {rec['nome']} ({rec['posicao']}) - {rec['clube']}\n")
            f.write(f"   Media: {rec['media']:.2f} | Preco: C$ {rec['preco']:.2f}\n")
            f.write(f"   Adversario: {rec['adversario']} ({rec['mando']})\n")
            f.write(f"   Custo-Beneficio: {rec['custo_beneficio']:.3f}\n\n")
        
        f.write("="*80 + "\n")
    
    print(f"\nRelatorio salvo em: {filename}")
    print()


def main():
    """Fluxo principal"""
    
    # Carregar dados
    dados_atletas, dados_clubes, dados_partidas = carregar_dados()
    
    if not dados_atletas:
        print("ERRO: Nao foi possivel carregar dados")
        return
    
    # Processar confrontos
    confrontos = processar_confrontos(dados_partidas, dados_clubes)
    
    # Analisar com confrontos
    recomendacoes = analisar_com_confrontos(dados_atletas, dados_clubes, confrontos)
    
    # Gerar recomendações
    gerar_recomendacoes(recomendacoes)
    
    print("="*80)
    print("ANALISE CONCLUIDA!")
    print("="*80)
    print()


if __name__ == "__main__":
    main()
