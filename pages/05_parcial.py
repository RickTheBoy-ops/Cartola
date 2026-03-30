#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
========================================================================
PARCIAL DA RODADA
========================================================================
Acompanhamento em tempo real da rodada em andamento.
Inspirада na funcionalidade 'Parcial' da Máquina do Cartola.

Funcionalidades:
  - Pontuação parcial do seu time escalado
  - Status dos jogos em andamento (ao vivo)
  - Comparativo: seu time vs. melhor escalação possível
  - Placar por jogador com eventos (gols, assistências, cartões)
  - Indicador se o capitão foi a melhor escolha
  - Ranking parcial por liga
========================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="Parcial da Rodada | Cartola AI",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Parcial da Rodada")
st.markdown("Acompanhe a pontuação do seu time em **tempo real** durante a rodada.")

POS_NOME = {1: 'GOL', 2: 'LAT', 3: 'ZAG', 4: 'MEI', 5: 'ATA', 6: 'TEC'}


@st.cache_data(ttl=60)  # Atualiza a cada 60s
def carregar_parcial():
    """Carrega dados parciais. Em produção, conectar à API do Cartola."""
    import glob
    csvs = glob.glob("data/parcial*.csv") + glob.glob("data/rodada_atual*.csv")
    if csvs:
        return pd.read_csv(csvs[0]), "real"

    # Demo: gerar parcial simulado
    rng = np.random.default_rng(int(datetime.now().timestamp()) % 1000)
    n_titulares = 12
    posicoes = [1, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5, 6]
    status_jogo = rng.choice(['Finalizado', 'Em andamento', 'Não iniciado'], n_titulares,
                             p=[0.4, 0.35, 0.25])
    pontos = []
    for s in status_jogo:
        if s == 'Finalizado':
            pontos.append(round(rng.normal(5.5, 3.0), 2))
        elif s == 'Em andamento':
            pontos.append(round(rng.normal(3.0, 2.0), 2))
        else:
            pontos.append(0.0)

    df = pd.DataFrame({
        'apelido': [f"Jogador_{i}" for i in range(1, n_titulares+1)],
        'posicao_id': posicoes,
        'clube_nome': [f"Clube_{rng.integers(1,21)}" for _ in range(n_titulares)],
        'adversario': [f"Clube_{rng.integers(1,21)}" for _ in range(n_titulares)],
        'pontos_parcial': [max(0.0, p) for p in pontos],
        'status_jogo': status_jogo,
        'gols': [rng.integers(0, 3) if s != 'Não iniciado' else 0 for s in status_jogo],
        'assistencias': [rng.integers(0, 2) if s != 'Não iniciado' else 0 for s in status_jogo],
        'cartao_amarelo': [rng.integers(0, 2) if s != 'Não iniciado' else 0 for s in status_jogo],
        'cartao_vermelho': [1 if rng.random() < 0.05 and s != 'Não iniciado' else 0 for s in status_jogo],
        'eh_capitao': [False]*n_titulares,
        'preco': rng.uniform(4.0, 35.0, n_titulares).round(2),
    })
    # Escolher capitão
    cap_idx = df['pontos_parcial'].idxmax()
    df.loc[rng.integers(0, n_titulares), 'eh_capitao'] = True
    return df, "sintetico"


df_parcial, fonte = carregar_parcial()

if fonte == 'sintetico':
    st.info("📡 Modo demonstração. Conecte a API do Cartola para dados em tempo real.")

# ── Auto-refresh ──────────────────────────────────────────────────────
col_refresh, col_timer = st.columns([1, 3])
with col_refresh:
    if st.button("🔄 Atualizar Agora"):
        st.cache_data.clear()
        st.rerun()
with col_timer:
    st.caption(f"Última atualização: {datetime.now().strftime('%H:%M:%S')}")

st.divider()

# ── Métricas do time ──────────────────────────────────────────────────
pontos_total = df_parcial['pontos_parcial'].sum()
capitao = df_parcial[df_parcial['eh_capitao'] == True]
if not capitao.empty:
    bonus_capitao = capitao.iloc[0]['pontos_parcial']  # capitão dobra
    pontos_com_cap = pontos_total + bonus_capitao
else:
    pontos_com_cap = pontos_total

finalizados = (df_parcial['status_jogo'] == 'Finalizado').sum()
em_campo = (df_parcial['status_jogo'] == 'Em andamento').sum()
nao_iniciados = (df_parcial['status_jogo'] == 'Não iniciado').sum()

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("⭐ Pontos Totais", f"{pontos_com_cap:.2f}", help="Inclui bônus do capitão")
m2.metric("✅ Finalizados", finalizados)
m3.metric("🔴 Em Campo", em_campo)
m4.metric("⏳ Aguardando", nao_iniciados)
m5.metric("🗂️ Jogadores", len(df_parcial))

# ── Tabs ──────────────────────────────────────────────────────────────
tab_time, tab_jogos, tab_capitao = st.tabs(["👥 Meu Time", "⚽ Jogos", "🎖️ Capitão"])

with tab_time:
    st.subheader("👥 Pontuação por Jogador")
    STATUS_COR = {'Finalizado': '🟢', 'Em andamento': '🔴', 'Não iniciado': '⚪'}

    for pos_id in [1, 2, 3, 4, 5, 6]:
        jogadores = df_parcial[df_parcial['posicao_id'] == pos_id]
        if jogadores.empty:
            continue
        pos_label = POS_NOME.get(pos_id, '?')
        cols = st.columns(len(jogadores))
        for col, (_, row) in zip(cols, jogadores.iterrows()):
            nome = row.get('apelido', 'N/A')
            pontos = row.get('pontos_parcial', 0)
            status = row.get('status_jogo', '')
            cap = " 🎖️" if row.get('eh_capitao') else ""
            icone = STATUS_COR.get(status, '⚪')
            bg = '#1a3a1a' if status == 'Finalizado' else '#3a1a1a' if status == 'Em andamento' else '#2a2a2a'
            col.markdown(f"""
            <div style='background:{bg};border-radius:10px;padding:12px;text-align:center;
            color:white;border:1px solid #555;margin:2px;'>
                <div style='font-size:10px;color:#aaa'>{pos_label} {icone}</div>
                <div style='font-size:14px;font-weight:700'>{nome}{cap}</div>
                <div style='font-size:20px;color:#FFD700;font-weight:700'>{pontos:.2f}</div>
                <div style='font-size:10px;color:#888'>{status}</div>
            </div>
            """, unsafe_allow_html=True)

with tab_jogos:
    st.subheader("⚽ Jogadores em Campo")
    cols_jogo = [c for c in ['apelido', 'posicao_id', 'clube_nome', 'adversario', 'pontos_parcial',
                              'gols', 'assistencias', 'cartao_amarelo', 'cartao_vermelho', 'status_jogo']
                 if c in df_parcial.columns]
    df_view = df_parcial[cols_jogo].sort_values('pontos_parcial', ascending=False)
    st.dataframe(df_view, use_container_width=True, hide_index=True)

with tab_capitao:
    st.subheader("🎖️ Análise do Capitão")
    melhor_possivel = df_parcial.loc[df_parcial['pontos_parcial'].idxmax()]
    cap_row = df_parcial[df_parcial['eh_capitao'] == True]

    col_cap, col_melhor = st.columns(2)
    with col_cap:
        st.markdown("**Seu Capitão**")
        if not cap_row.empty:
            r = cap_row.iloc[0]
            st.metric(r.get('apelido', 'N/A'), f"{r['pontos_parcial']:.2f} pts")
            st.caption(f"{POS_NOME.get(int(r.get('posicao_id', 0)), '?')} — {r.get('clube_nome', '')}")
        else:
            st.warning("Capitão não definido")
    with col_melhor:
        st.markdown("**Melhor Opção (hindsight)**")
        st.metric(melhor_possivel.get('apelido', 'N/A'), f"{melhor_possivel['pontos_parcial']:.2f} pts")
        st.caption(f"{POS_NOME.get(int(melhor_possivel.get('posicao_id', 0)), '?')} — {melhor_possivel.get('clube_nome', '')}")

    if not cap_row.empty:
        diff = melhor_possivel['pontos_parcial'] - cap_row.iloc[0]['pontos_parcial']
        if diff <= 0:
            st.success("✅ Ótima escolha! Seu capitão é o melhor da rodada até agora.")
        elif diff < 3:
            st.warning(f"⚠️ Diferença pequena: {diff:.2f} pts para o melhor.")
        else:
            st.error(f"❌ Diferença de {diff:.2f} pts. Próxima rodada: analise melhor o capitão no Olheiro.")
