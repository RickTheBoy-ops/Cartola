#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
========================================================================
CARTOLA FC - SISTEMA INTEGRADO DE ESCALAÇÃO INTELIGENTE
========================================================================
Sistema completo que:
- Coleta dados atualizados da API do Cartola
- Pergunta orçamento e objetivo (pontos ou valorização)
- Analisa com IA (Perplexity)
- Gera escalação otimizada
========================================================================
"""

import requests
import json
from typing import Dict, List, Tuple
from datetime import datetime
import os

# ========================================================================
# CONFIGURAÇÃO
# ========================================================================

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "")

# ========================================================================
# 1. COLETA DE DADOS DA API
# ========================================================================

def carregar_dados_cartola():
    """Carrega dados das APIs do Cartola"""
    
    API_ATLETAS = "https://api.cartola.globo.com/atletas/mercado"
    API_CLUBES = "https://api.cartola.globo.com/clubes"
    API_PARTIDAS = "https://api.cartola.globo.com/partidas"
    
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    print("\n" + "="*80)
    print("CARTOLA FC - SISTEMA INTEGRADO DE ESCALACAO INTELIGENTE")
    print("="*80)
    print(f"\nHorario: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print("\nCarregando dados atualizados da API do Cartola...")
    
    try:
        r1 = requests.get(API_ATLETAS, headers=HEADERS, timeout=30)
        r2 = requests.get(API_CLUBES, headers=HEADERS, timeout=30)
        r3 = requests.get(API_PARTIDAS, headers=HEADERS, timeout=30)
        
        print("Dados carregados com sucesso!\n")
        return r1.json(), r2.json(), r3.json()
    
    except Exception as e:
        print(f"ERRO ao carregar dados: {e}\n")
        return None, None, None


def processar_atletas(dados_atletas, dados_clubes):
    """Processa e estrutura dados dos atletas"""
    
    # Mapear clubes
    clubes_dict = {int(k): v.get('nome', 'Desconhecido') for k, v in dados_clubes.items()}
    
    # Processar atletas
    atletas = dados_atletas.get('atletas', [])
    if isinstance(atletas, dict):
        atletas = list(atletas.values())
    
    # Filtrar apenas jogadores prováveis (status_id == 7)
    atletas_provaveis = []
    
    POS_MAP = {1: "Goleiro", 2: "Lateral", 3: "Zagueiro", 4: "Meia", 5: "Atacante", 6: "Técnico"}
    
    for atleta in atletas:
        if atleta.get('status_id') == 7:  # Provável
            atleta_info = {
                'id': atleta.get('atleta_id'),
                'nome': atleta.get('apelido', 'Desconhecido'),
                'posicao_id': atleta.get('posicao_id'),
                'posicao': POS_MAP.get(atleta.get('posicao_id'), 'Desconhecido'),
                'clube_id': atleta.get('clube_id'),
                'clube': clubes_dict.get(atleta.get('clube_id'), 'Desconhecido'),
                'preco': atleta.get('preco_num', 0),
                'media': atleta.get('media_num', 0),
                'pontos_ultima': atleta.get('pontos_num', 0),
                'jogos': atleta.get('jogos_num', 0),
                'variacao': atleta.get('variacao_num', 0)
            }
            
            # Calcular custo-benefício
            if atleta_info['preco'] > 0:
                atleta_info['custo_beneficio'] = atleta_info['media'] / atleta_info['preco']
            else:
                atleta_info['custo_beneficio'] = 0
            
            atletas_provaveis.append(atleta_info)
    
    return atletas_provaveis


# ========================================================================
# 2. INTERAÇÃO COM USUÁRIO
# ========================================================================

def perguntar_usuario():
    """Pergunta orçamento e objetivo ao usuário"""
    
    print("="*80)
    print("CONFIGURACAO DA ESCALACAO")
    print("="*80)
    print()
    
    # Perguntar orçamento
    while True:
        try:
            orcamento_input = input("Qual seu orcamento disponivel? (Ex: 100 ou C$100): ")
            orcamento_input = orcamento_input.replace('C$', '').replace('$', '').strip()
            orcamento = float(orcamento_input)
            
            if orcamento <= 0:
                print("ERRO: Orcamento deve ser maior que zero!\n")
                continue
            
            print(f"\nOrcamento: C$ {orcamento:.2f}")
            break
        except ValueError:
            print("ERRO: Digite um valor valido!\n")
    
    print()
    
    # Perguntar objetivo
    while True:
        print("Qual seu objetivo?")
        print("  1 - FAZER PONTOS (maximizar media)")
        print("  2 - VALORIZACAO (buscar jogadores baratos que podem valorizar)")
        print()
        
        objetivo_input = input("Escolha (1 ou 2): ").strip()
        
        if objetivo_input == '1':
            objetivo = 'pontos'
            print("\nObjetivo: FAZER PONTOS\n")
            break
        elif objetivo_input == '2':
            objetivo = 'valorizacao'
            print("\nObjetivo: VALORIZACAO\n")
            break
        else:
            print("ERRO: Escolha 1 ou 2!\n")
    
    return orcamento, objetivo


# ========================================================================
# 3. ANÁLISE COM IA (PERPLEXITY)
# ========================================================================

def analisar_com_ia(atletas_top, orcamento, objetivo):
    """Analisa com IA Perplexity"""
    
    print("="*80)
    print("ANALISE COM IA")
    print("="*80)
    print("\nConsultando IA Perplexity... (aguarde 30-60s)\n")
    
    # Preparar contexto
    contexto = f"Orcamento: C$ {orcamento:.2f}\n"
    contexto += f"Objetivo: {'FAZER PONTOS (maximizar pontuacao)' if objetivo == 'pontos' else 'VALORIZACAO (jogadores baratos com potencial)'}\n\n"
    contexto += "Top jogadores disponiveis por posicao:\n\n"
    
    # Agrupar por posição
    por_posicao = {}
    for atleta in atletas_top:
        pos = atleta['posicao']
        if pos not in por_posicao:
            por_posicao[pos] = []
        por_posicao[pos].append(atleta)
    
    for pos, jogadores in por_posicao.items():
        contexto += f"{pos}:\n"
        for j in jogadores[:5]:  # Top 5
            contexto += f"  - {j['nome']} ({j['clube']}) - Media: {j['media']:.1f}, Preco: C${j['preco']:.1f}\n"
        contexto += "\n"
    
    if not PERPLEXITY_API_KEY:
        avg_media = sum([a['media'] for a in atletas_top]) / max(1, len(atletas_top))
        return (
            "ANÁLISE AUTOMÁTICA (Sem IA):\n"
            "—\n"
            f"MÉDIA DO TIME PROJETADA: {avg_media:.2f}\n"
            "NOTA: 7/10\n"
            "SUGESTÃO: Verifique notícias de última hora manualmente.\n"
            "—\n"
            "⚠️ Configure PERPLEXITY_API_KEY em variáveis de ambiente para ativar a IA."
        )
    # Prompt para IA
    system_prompt = """Voce é um especialista em Cartola FC e análise de futebol brasileiro.
Forneça recomendações objetivas e práticas baseadas em:
- Forma recente dos jogadores
- Confrontos favoráveis
- Lesões e suspensões
- Relação custo-benefício"""
    
    if objetivo == 'pontos':
        user_prompt = f"""{contexto}

Com base nesses dados e no orcamento de C$ {orcamento:.2f}, me ajude a montar uma escalacao para FAZER PONTOS.

Forneça:
1. Top 3 jogadores IMPERDÍVEIS para escalar (considerando media e preco)
2. Recomendacao de capitao
3. Confrontos favoraveis para pontuar
4. Jogadores a evitar (lesoes, suspensos, ma forma)
5. Dica final de estratégia

Seja especifico com nomes de jogadores."""
    else:
        user_prompt = f"""{contexto}

Com base nesses dados e no orcamento de C$ {orcamento:.2f}, me ajude a montar uma escalacao para VALORIZACAO.

Forneça:
1. Top 3 jogadores BARATOS com POTENCIAL de valorizacao
2. Quais posicoes investir menos (zagueiros baratos, etc)
3. Jogadores que podem pontuar bem e valorizar
4. Confrontos que favorecem times mais fracos
5. Dica final de estratégia

Seja especifico com nomes de jogadores."""
    
    # Chamar API
    try:
        url = "https://api.perplexity.ai/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
            "Content-Type": "application/json"
        }
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        payload = {
            "model": "sonar",
            "messages": messages,
            "max_tokens": 1500,
            "temperature": 0.4,
            "top_p": 0.9
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=90)
        
        if response.status_code == 200:
            data = response.json()
            analise = data['choices'][0]['message']['content']
            print("Analise concluida!\n")
            return analise
        else:
            print(f"ERRO na API: {response.status_code} - {response.text}\n")
            return "Analise com IA indisponivel no momento."
    
    except Exception as e:
        print(f"ERRO ao consultar IA: {str(e)}\n")
        return "Analise com IA indisponivel no momento."


# ========================================================================
# 4. OTIMIZAÇÃO DE ESCALAÇÃO
# ========================================================================

def otimizar_escalacao(atletas, orcamento, objetivo):
    """Gera escalação otimizada"""
    
    print("="*80)
    print("OTIMIZANDO ESCALACAO")
    print("="*80)
    print()
    
    # Separar por posição
    goleiros = [a for a in atletas if a['posicao_id'] == 1]
    laterais = [a for a in atletas if a['posicao_id'] == 2]
    zagueiros = [a for a in atletas if a['posicao_id'] == 3]
    meias = [a for a in atletas if a['posicao_id'] == 4]
    atacantes = [a for a in atletas if a['posicao_id'] == 5]
    tecnicos = [a for a in atletas if a['posicao_id'] == 6]
    
    # Ordenar por métrica escolhida
    if objetivo == 'pontos':
        # Ordenar por média (priorizar pontos)
        goleiros.sort(key=lambda x: x['media'], reverse=True)
        laterais.sort(key=lambda x: x['media'], reverse=True)
        zagueiros.sort(key=lambda x: x['media'], reverse=True)
        meias.sort(key=lambda x: x['media'], reverse=True)
        atacantes.sort(key=lambda x: x['media'], reverse=True)
        tecnicos.sort(key=lambda x: x['media'], reverse=True)
    else:
        # Ordenar por custo-benefício (valorização)
        goleiros.sort(key=lambda x: x['custo_beneficio'], reverse=True)
        laterais.sort(key=lambda x: x['custo_beneficio'], reverse=True)
        zagueiros.sort(key=lambda x: x['custo_beneficio'], reverse=True)
        meias.sort(key=lambda x: x['custo_beneficio'], reverse=True)
        atacantes.sort(key=lambda x: x['custo_beneficio'], reverse=True)
        tecnicos.sort(key=lambda x: x['custo_beneficio'], reverse=True)
    
    # Montar escalação (formação 3-4-3 ou variações)
    escalacao = []
    gasto = 0
    
    # 1 Goleiro
    if goleiros and gasto + goleiros[0]['preco'] <= orcamento:
        escalacao.append(goleiros[0])
        gasto += goleiros[0]['preco']
    
    # 2 Laterais
    for i in range(2):
        if i < len(laterais) and gasto + laterais[i]['preco'] <= orcamento:
            escalacao.append(laterais[i])
            gasto += laterais[i]['preco']
    
    # 3 Zagueiros
    for i in range(3):
        if i < len(zagueiros) and gasto + zagueiros[i]['preco'] <= orcamento:
            escalacao.append(zagueiros[i])
            gasto += zagueiros[i]['preco']
    
    # 4 Meias
    for i in range(4):
        if i < len(meias) and gasto + meias[i]['preco'] <= orcamento:
            escalacao.append(meias[i])
            gasto += meias[i]['preco']
    
    # 3 Atacantes
    for i in range(3):
        if i < len(atacantes) and gasto + atacantes[i]['preco'] <= orcamento:
            escalacao.append(atacantes[i])
            gasto += atacantes[i]['preco']
    
    # 1 Técnico
    if tecnicos and gasto + tecnicos[0]['preco'] <= orcamento:
        escalacao.append(tecnicos[0])
        gasto += tecnicos[0]['preco']
    
    # Selecionar capitão (maior média)
    capitao = None
    if escalacao:
        capitao = max(escalacao, key=lambda x: x['media'])
    
    return escalacao, capitao, gasto


# ========================================================================
# 5. GERAÇÃO DE RELATÓRIO
# ========================================================================

def gerar_relatorio(escalacao, capitao, gasto, orcamento, objetivo, analise_ia):
    """Gera relatório final"""
    
    print("\n" + "="*80)
    print("ESCALACAO OTIMIZADA - CARTOLA FC")
    print("="*80)
    print()
    
    print(f"Objetivo: {'FAZER PONTOS' if objetivo == 'pontos' else 'VALORIZACAO'}")
    print(f"Orcamento: C$ {orcamento:.2f}")
    print(f"Gasto: C$ {gasto:.2f}")
    print(f"Sobra: C$ {orcamento - gasto:.2f}")
    print()
    
    # Agrupar por posição
    por_posicao = {}
    for atleta in escalacao:
        pos = atleta['posicao']
        if pos not in por_posicao:
            por_posicao[pos] = []
        por_posicao[pos].append(atleta)
    
    # Exibir escalação
    print("-"*80)
    print("ESCALACAO:")
    print("-"*80)
    
    ordem_posicoes = ['Goleiro', 'Lateral', 'Zagueiro', 'Meia', 'Atacante', 'Técnico']
    
    media_total = 0
    
    for pos in ordem_posicoes:
        if pos in por_posicao:
            print(f"\n{pos.upper()}:")
            for atleta in por_posicao[pos]:
                eh_capitao = " (C)" if capitao and atleta['id'] == capitao['id'] else ""
                print(f"  - {atleta['nome']:20} ({atleta['clube']:15}) Media: {atleta['media']:5.2f} | Preco: C$ {atleta['preco']:6.2f}{eh_capitao}")
                media_total += atleta['media']
    
    print()
    print("-"*80)
    print(f"MEDIA TOTAL ESPERADA: {media_total:.2f} pontos")
    
    if capitao:
        print(f"CAPITAO: {capitao['nome']} (dobra a pontuacao)")
        print(f"PONTUACAO ESPERADA COM CAPITAO: {media_total + capitao['media']:.2f} pontos")
    
    print("-"*80)
    print()
    
    # Análise IA
    print("="*80)
    print("ANALISE DA IA")
    print("="*80)
    print()
    print(analise_ia)
    print()
    print("="*80)
    print()
    
    # Salvar em arquivo
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"escalacao_otimizada_{timestamp}.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("ESCALACAO OTIMIZADA - CARTOLA FC\n")
        f.write("="*80 + "\n\n")
        f.write(f"Data/Hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
        f.write(f"Objetivo: {'FAZER PONTOS' if objetivo == 'pontos' else 'VALORIZACAO'}\n")
        f.write(f"Orcamento: C$ {orcamento:.2f}\n")
        f.write(f"Gasto: C$ {gasto:.2f}\n")
        f.write(f"Sobra: C$ {orcamento - gasto:.2f}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("ESCALACAO:\n")
        f.write("-"*80 + "\n\n")
        
        for pos in ordem_posicoes:
            if pos in por_posicao:
                f.write(f"{pos.upper()}:\n")
                for atleta in por_posicao[pos]:
                    eh_capitao = " (C)" if capitao and atleta['id'] == capitao['id'] else ""
                    f.write(f"  - {atleta['nome']:20} ({atleta['clube']:15}) Media: {atleta['media']:5.2f} | Preco: C$ {atleta['preco']:6.2f}{eh_capitao}\n")
                f.write("\n")
        
        f.write("-"*80 + "\n")
        f.write(f"MEDIA TOTAL ESPERADA: {media_total:.2f} pontos\n")
        if capitao:
            f.write(f"CAPITAO: {capitao['nome']} (dobra a pontuacao)\n")
            f.write(f"PONTUACAO ESPERADA COM CAPITAO: {media_total + capitao['media']:.2f} pontos\n")
        f.write("-"*80 + "\n\n")
        
        f.write("="*80 + "\n")
        f.write("ANALISE DA IA\n")
        f.write("="*80 + "\n\n")
        f.write(analise_ia)
        f.write("\n\n")
        f.write("="*80 + "\n")
    
    print(f"Relatorio salvo em: {filename}")
    print()


# ========================================================================
# MAIN
# ========================================================================

def main():
    """Fluxo principal"""
    
    # 1. Carregar dados da API
    dados_atletas, dados_clubes, dados_partidas = carregar_dados_cartola()
    
    if not dados_atletas:
        print("ERRO: Nao foi possivel carregar dados do Cartola FC")
        return
    
    atletas = processar_atletas(dados_atletas, dados_clubes)
    
    print(f"Total de jogadores provaveis: {len(atletas)}")
    print()
    
    # 2. Perguntar orçamento e objetivo
    orcamento, objetivo = perguntar_usuario()
    
    # 3. Analisar com IA
    # Pegar top jogadores para análise IA
    atletas_por_media = sorted(atletas, key=lambda x: x['media'], reverse=True)
    top_por_posicao = {}
    for atleta in atletas_por_media:
        pos = atleta['posicao']
        if pos not in top_por_posicao:
            top_por_posicao[pos] = []
        if len(top_por_posicao[pos]) < 10:
            top_por_posicao[pos].append(atleta)
    
    top_atletas = []
    for jogadores in top_por_posicao.values():
        top_atletas.extend(jogadores)
    
    analise_ia = analisar_com_ia(top_atletas, orcamento, objetivo)
    
    # 4. Otimizar escalação
    escalacao, capitao, gasto = otimizar_escalacao(atletas, orcamento, objetivo)
    
    if not escalacao:
        print("ERRO: Nao foi possivel gerar escalacao com o orcamento disponivel")
        return
    
    # 5. Gerar relatório
    gerar_relatorio(escalacao, capitao, gasto, orcamento, objetivo, analise_ia)
    
    print("="*80)
    print("PROCESSO CONCLUIDO!")
    print("="*80)


if __name__ == "__main__":
    main()
