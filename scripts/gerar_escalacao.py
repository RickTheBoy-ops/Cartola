#!/usr/bin/env python3
"""
Gerar Escalação 4-3-3 com orçamento definido
Busca dados da API do Cartola FC e otimiza o time
"""

import sys
import os
import json
import requests
import pandas as pd
from itertools import combinations

# ──────────────────────────────────────────
# Configurações
# ──────────────────────────────────────────
ORCAMENTO    = 112.0        # cartoletas
FORMACAO     = "4-3-3"      # GOL-1, LAT-2, ZAG-2, MEI-3, ATA-3, TEC-1 (12 total)
MAX_CLUBE    = 5            # máximo jogadores do mesmo clube

BASE_URL = "https://api.cartolafc.globo.com"

POSICAO_NOME = {1:'GOL', 2:'LAT', 3:'ZAG', 4:'MEI', 5:'ATA', 6:'TEC'}
# Formação 4-3-3:  1 GOL, 2 LAT, 2 ZAG, 3 MEI, 3 ATA, 1 TEC
SLOTS_433 = {1:1, 2:2, 3:2, 4:3, 5:3, 6:1}

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
    'Accept': 'application/json',
}

def fetch(endpoint):
    url = f"{BASE_URL}{endpoint}"
    print(f"  → GET {url}")
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()

def main():
    print("\n" + "="*60)
    print("  CARTOLA FC OPTIMIZER - 4-3-3 | Orçamento: C$ 112")
    print("="*60 + "\n")

    # ── 1. Mercado status ──────────────────────────────────────
    print("[1/4] Buscando status do mercado...")
    status = fetch("/mercado/status")
    rodada   = status.get('rodada_atual', status.get('rodada', '?'))
    temporada = status.get('temporada', 2026)
    rodada_label = f"Rodada {rodada} / {temporada}"
    print(f"      ✅ {rodada_label}")

    # ── 2. Atletas do mercado ──────────────────────────────────
    print("\n[2/4] Buscando atletas disponíveis no mercado...")
    mercado = fetch("/atletas/mercado")

    atletas_raw = mercado.get('atletas', [])
    clubes_map  = {int(k): v.get('nome', k) for k, v in mercado.get('clubes', {}).items()}
    
    print(f"      ✅ {len(atletas_raw)} atletas recebidos")

    # ── 3. Partidas da rodada ──────────────────────────────────
    print(f"\n[3/4] Buscando partidas da rodada {rodada}...")
    try:
        partidas_data = fetch(f"/partidas/{rodada}")
        partidas = partidas_data.get('partidas', [])
        print(f"      ✅ {len(partidas)} partidas encontradas")
        # Montar set de clubes com jogo
        clubes_com_jogo = set()
        for p in partidas:
            clubes_com_jogo.add(p.get('clube_casa_id'))
            clubes_com_jogo.add(p.get('clube_visitante_id'))
        print(f"      ✅ {len(clubes_com_jogo)} clubes com jogo confirmado")
    except Exception as e:
        print(f"      ⚠️ Partidas não disponíveis: {e}")
        partidas = []
        clubes_com_jogo = None

    # ── 4. Processar atletas ───────────────────────────────────
    print("\n[4/4] Processando e otimizando escalação 4-3-3...")

    rows = []
    for a in atletas_raw:
        pos_id    = a.get('posicao_id', 0)
        clube_id  = a.get('clube_id', 0)
        status_id = a.get('status_id', 7)

        # Apenas posições necessárias
        if pos_id not in SLOTS_433:
            continue
        # Atletas prováveis ou titular (status_id 2, 3, 4, 5, 6, 7)
        # 7 = dúvida, 2 = contundido (excluir), 3 = suspenso (excluir)
        if status_id in [2, 3]:  # Contundido ou Suspenso
            continue
        # Filtrar por clube com jogo (se disponível)
        if clubes_com_jogo and clube_id not in clubes_com_jogo:
            continue

        scout = a.get('scout', {}) or {}
        media = a.get('media_num', 0) or 0
        preco = a.get('preco_num', 0) or 0

        if preco < 1.0:
            continue

        # Score simples: média ponderada com eficiência (pontos/cartoleta)
        eficiencia = media / preco if preco > 0 else 0
        score = (media * 0.6) + (eficiencia * 3.0)

        rows.append({
            'atleta_id': a.get('atleta_id'),
            'apelido':   a.get('apelido', a.get('nome', '?')),
            'posicao_id': pos_id,
            'posicao':   POSICAO_NOME.get(pos_id, '?'),
            'clube_id':  clube_id,
            'clube':     clubes_map.get(clube_id, str(clube_id)),
            'preco':     preco,
            'media':     media,
            'score':     score,
            'status_id': status_id,
            'jogos_num': a.get('jogos_num', 0) or 0,
            'variacao':  a.get('variacao_num', 0) or 0,
        })

    df = pd.DataFrame(rows)
    print(f"      → {len(df)} atletas elegíveis após filtros")

    if df.empty:
        print("❌ Nenhum atleta elegível encontrado!")
        return

    # ── 5. Otimização Greedy (relação pontos/custo) ────────────
    team = []
    custo_total = 0.0
    count_pos   = {pos: 0 for pos in SLOTS_433}
    count_clube = {}

    # Ordenar por score desc
    df_sorted = df.sort_values('score', ascending=False).reset_index(drop=True)

    for _, row in df_sorted.iterrows():
        pos_id   = int(row['posicao_id'])
        clube_id = int(row['clube_id'])
        preco    = float(row['preco'])

        # Checar slot disponível
        if count_pos.get(pos_id, 0) >= SLOTS_433.get(pos_id, 0):
            continue
        # Checar orçamento
        if custo_total + preco > ORCAMENTO:
            continue
        # Checar max por clube
        if count_clube.get(clube_id, 0) >= MAX_CLUBE:
            continue

        team.append(row.to_dict())
        custo_total += preco
        count_pos[pos_id] += 1
        count_clube[clube_id] = count_clube.get(clube_id, 0) + 1

        if sum(count_pos.values()) == 12:
            break

    # ── Verificar se time está completo ───────────────────────
    total_slots = sum(SLOTS_433.values())
    n_selecionados = sum(count_pos.values())
    
    if n_selecionados < total_slots:
        print(f"\n  ⚠️  Time incompleto ({n_selecionados}/{total_slots}). Tentando relaxar filtros...")
        # Tentar incluir todos atletas (sem filtro de clube com jogo)
        df2 = pd.DataFrame(rows if rows else [])
        team = []
        custo_total = 0.0
        count_pos   = {pos: 0 for pos in SLOTS_433}
        count_clube = {}
        df2_sorted = df2.sort_values('score', ascending=False).reset_index(drop=True)
        for _, row in df2_sorted.iterrows():
            pos_id   = int(row['posicao_id'])
            clube_id = int(row['clube_id'])
            preco    = float(row['preco'])
            if count_pos.get(pos_id, 0) >= SLOTS_433.get(pos_id, 0): continue
            if custo_total + preco > ORCAMENTO: continue
            if count_clube.get(clube_id, 0) >= MAX_CLUBE: continue
            team.append(row.to_dict())
            custo_total += preco
            count_pos[pos_id] += 1
            count_clube[clube_id] = count_clube.get(clube_id, 0) + 1
            if sum(count_pos.values()) == 12: break

    # ── 6. Exibir resultado ────────────────────────────────────
    team_df = pd.DataFrame(team)

    print("\n" + "="*60)
    print(f"  🏆 ESCALAÇÃO 4-3-3 — {rodada_label}")
    print("="*60)

    POS_ORDER = [1, 3, 2, 4, 5, 6]  # GOL, ZAG, LAT, MEI, ATA, TEC
    POS_LABEL = {
        1: '⛔ GOLEIRO',
        2: '🛡️  LATERAIS',
        3: '🛡️  ZAGUEIROS',
        4: '⚙️  MEIAS',
        5: '🔥 ATACANTES',
        6: '📋 TÉCNICO'
    }

    total_pontos = 0
    for pos_id in POS_ORDER:
        jogadores_pos = [p for p in team if p['posicao_id'] == pos_id]
        if not jogadores_pos:
            continue
        print(f"\n  {POS_LABEL.get(pos_id, '?')}")
        print(f"  {'JOGADOR':<22} {'CLUBE':<18} {'PREÇO':>7} {'MÉDIA':>7} {'SCORE':>7}")
        print(f"  {'-'*62}")
        for p in sorted(jogadores_pos, key=lambda x: -x['score']):
            nome  = p['apelido'][:21]
            clube = p['clube'][:17]
            preco = p['preco']
            media = p['media']
            score = p['score']
            total_pontos += media
            print(f"  {nome:<22} {clube:<18} C${preco:>5.1f} {media:>7.2f} {score:>7.2f}")

    print(f"\n{'='*60}")
    print(f"  💰 CUSTO TOTAL:      C$ {custo_total:.1f}  (orçamento: C$ {ORCAMENTO:.1f})")
    print(f"  📊 MÉDIA ESPERADA:   {total_pontos:.1f} pts")
    print(f"  💵 SALDO RESTANTE:   C$ {ORCAMENTO - custo_total:.1f}")
    if clubes_com_jogo:
        print(f"  🏟️  PARTIDAS NA RODADA: {len(partidas)}")
    print(f"{'='*60}\n")

    # ── 7. Exibir partidas da rodada ───────────────────────────
    if partidas:
        print("\n📅 JOGOS DA PRÓXIMA RODADA:")
        print(f"  {'MANDANTE':<22} {'VISITANTE':<22} {'DATA':<20}")
        print(f"  {'-'*65}")
        for p in partidas:
            casa      = clubes_map.get(p.get('clube_casa_id', 0), '?')[:21]
            visit     = clubes_map.get(p.get('clube_visitante_id', 0), '?')[:21]
            data_hora = p.get('partida_data', '?')[:19] if p.get('partida_data') else '?'
            print(f"  {casa:<22} {visit:<22} {data_hora}")

    # ── 8. Salvar resultado ────────────────────────────────────
    out_path = os.path.join(os.path.dirname(__file__), 'data', 'processed', 'escalacao_433.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    resultado = {
        'rodada': rodada,
        'temporada': temporada,
        'formacao': FORMACAO,
        'orcamento': ORCAMENTO,
        'custo_total': round(custo_total, 2),
        'saldo_restante': round(ORCAMENTO - custo_total, 2),
        'media_total': round(total_pontos, 2),
        'time': [
            {
                'posicao': p['posicao'],
                'apelido': p['apelido'],
                'clube': p['clube'],
                'preco': p['preco'],
                'media': p['media'],
                'score': round(p['score'], 2),
            }
            for p in team
        ],
        'partidas': [
            {
                'mandante': clubes_map.get(p.get('clube_casa_id', 0), '?'),
                'visitante': clubes_map.get(p.get('clube_visitante_id', 0), '?'),
                'data': p.get('partida_data', '?'),
            }
            for p in partidas
        ]
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)
    print(f"\n  ✅ Resultado salvo em: {out_path}\n")

    return resultado

if __name__ == '__main__':
    resultado = main()
