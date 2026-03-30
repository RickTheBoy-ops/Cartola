#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
========================================================================
OLHEIRO — Análise Individual de Jogadores
========================================================================
Inspirада na funcionalidade 'Olheiro' da Máquina do Cartola.

Funcionalidades:
  - Ranking geral de jogadores por mega_score
  - Filtros por posição, clube, preço, status
  - Gráfico de radar por jogador (perfil estatístico)
  - Comparador lado a lado (até 3 jogadores)
  - Análise de custo-benefício (score/preço)
  - Histórico de pontuação (quando disponível)
========================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.feature_engineering_v2 import FeatureEngineeringV2

st.set_page_config(
    page_title="Olheiro | Cartola AI",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Olheiro — Análise de Jogadores")
st.markdown("Encontre os melhores custo-benefício da rodada com análise estatística por posição.")

POS_NOME = {1: 'GOL', 2: 'LAT', 3: 'ZAG', 4: 'MEI', 5: 'ATA', 6: 'TEC'}
POS_COR = {1: '#f1c40f', 2: '#2ecc71', 3: '#3498db', 4: '#9b59b6', 5: '#e74c3c', 6: '#e67e22'}


@st.cache_data(ttl=300)
def carregar_e_processar():
    import glob
    csvs = glob.glob("data/jogadores*.csv") + glob.glob("data/atletas*.csv")
    if csvs:
        df_raw = pd.read_csv(csvs[0])
    else:
        rng = np.random.default_rng(42)
        n = 150
        posicoes = ([1]*15 + [2]*25 + [3]*25 + [4]*45 + [5]*30 + [6]*10)
        clubes = [f"Clube_{i}" for i in range(1, 21)]
        df_raw = pd.DataFrame({
            'atleta_id': range(1, n+1),
            'apelido': [f"Jogador_{i}" for i in range(1, n+1)],
            'posicao_id': posicoes[:n],
            'clube_id': rng.integers(1, 21, n),
            'clube_nome': [clubes[i % 20] for i in range(n)],
            'preco': rng.uniform(4.0, 40.0, n).round(2),
            'media': rng.uniform(1.5, 11.0, n).round(2),
            'pontos_ultimas_5': rng.uniform(0.5, 13.0, n).round(2),
            'jogos': rng.integers(2, 25, n),
            'minutos_jogados': rng.integers(180, 2000, n),
            'status': rng.choice(['Provável', 'Provável', 'Provável', 'Dúvida', 'Suspenso'], n),
            'variacao_preco': rng.uniform(-3.0, 5.0, n).round(2),
        })
    fe = FeatureEngineeringV2()
    try:
        df = fe.engineer_features(df_raw)
    except Exception:
        df = df_raw.copy()
        df['mega_score'] = df.get('media', 0) * 10
    df['pos_nome'] = df['posicao_id'].map(POS_NOME)
    df['custo_beneficio'] = (df['mega_score'] / df['preco'].replace(0, 1)).round(2)
    return df


df = carregar_e_processar()

# ── Filtros ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔎 Filtros")
    pos_filter = st.multiselect(
        "Posição", list(POS_NOME.values()), default=list(POS_NOME.values())
    )
    preco_min, preco_max = float(df['preco'].min()), float(df['preco'].max())
    preco_range = st.slider("Faixa de Preço (C$)", preco_min, preco_max, (preco_min, preco_max))
    status_filter = st.multiselect(
        "Status", df['status'].unique().tolist() if 'status' in df.columns else ['Provável'],
        default=['Provável']
    ) if 'status' in df.columns else None
    min_jogos = st.slider("Mínimo de jogos", 0, 20, 3)
    ordenar_por = st.selectbox("Ordenar por", ['mega_score', 'custo_beneficio', 'media', 'pontos_ultimas_5', 'preco'])

# ── Aplicar filtros ───────────────────────────────────────────────────
df_filtered = df[df['pos_nome'].isin(pos_filter)]
df_filtered = df_filtered[df_filtered['preco'].between(*preco_range)]
if 'jogos' in df_filtered.columns:
    df_filtered = df_filtered[df_filtered['jogos'] >= min_jogos]
if status_filter and 'status' in df_filtered.columns:
    df_filtered = df_filtered[df_filtered['status'].isin(status_filter)]
df_filtered = df_filtered.sort_values(ordenar_por, ascending=False).reset_index(drop=True)

# ── Métricas resumo ───────────────────────────────────────────────────
st.divider()
cols_met = st.columns(5)
cols_met[0].metric("👥 Jogadores", len(df_filtered))
cols_met[1].metric("⭐ Mega Score Médio", f"{df_filtered['mega_score'].mean():.1f}")
cols_met[2].metric("💰 Preço Médio", f"C$ {df_filtered['preco'].mean():.1f}")
cols_met[3].metric("📊 Melhor C/B", f"{df_filtered['custo_beneficio'].max():.2f}")
cols_met[4].metric("🏆 Top Score", f"{df_filtered['mega_score'].max():.1f}")

# ── Tabs ──────────────────────────────────────────────────────────────
tab_rank, tab_comparar, tab_custo = st.tabs(["🏆 Ranking", "⚖️ Comparador", "💡 Custo-Benefício"])

with tab_rank:
    st.subheader(f"🏆 Ranking — {len(df_filtered)} jogadores")

    # Top 3 destaque
    top3 = df_filtered.head(3)
    cols_top = st.columns(3)
    medalhas = ["🥇", "🥈", "🥉"]
    for i, (col, (_, row)) in enumerate(zip(cols_top, top3.iterrows())):
        nome = row.get('apelido', 'N/A')
        pos = row.get('pos_nome', '?')
        score = row.get('mega_score', 0)
        preco = row.get('preco', 0)
        media = row.get('media', 0)
        col.markdown(f"""
        <div style='background:linear-gradient(135deg,#1e3a1e,#2d5a2d);border-radius:12px;
        padding:16px;text-align:center;color:white;border:1px solid #4a9a4a;'>
            <div style='font-size:28px'>{medalhas[i]}</div>
            <div style='font-size:16px;font-weight:700'>{nome}</div>
            <div style='color:#aaa;font-size:12px'>{pos}</div>
            <div style='font-size:22px;color:#FFD700;font-weight:700'>{score:.1f}</div>
            <div style='color:#8af0a0;font-size:13px'>C$ {preco:.1f} | Média {media:.1f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")
    cols_show = [c for c in ['apelido', 'pos_nome', 'clube_nome', 'mega_score', 'custo_beneficio', 'media', 'pontos_ultimas_5', 'preco', 'jogos', 'status', 'variacao_preco'] if c in df_filtered.columns]
    st.dataframe(df_filtered[cols_show], use_container_width=True, hide_index=True)

with tab_comparar:
    st.subheader("⚖️ Comparar Jogadores")
    nomes_disp = df_filtered['apelido'].tolist() if 'apelido' in df_filtered.columns else df_filtered.index.tolist()
    selecionados = st.multiselect("Selecione até 3 jogadores para comparar", nomes_disp, max_selections=3)

    if len(selecionados) >= 2:
        df_comp = df_filtered[df_filtered['apelido'].isin(selecionados)]
        features_comp = [f for f in ['mega_score', 'media', 'pontos_ultimas_5', 'custo_beneficio', 'jogos', 'preco'] if f in df_comp.columns]

        try:
            import plotly.graph_objects as go
            fig = go.Figure()
            cores = ['#FFD700', '#00CED1', '#FF69B4']
            for i, (_, row) in enumerate(df_comp.iterrows()):
                vals = [row.get(f, 0) for f in features_comp]
                # Normalizar 0-100 para radar
                maxvals = [df_filtered[f].max() for f in features_comp]
                vals_norm = [v / m * 100 if m > 0 else 0 for v, m in zip(vals, maxvals)]
                fig.add_trace(go.Scatterpolar(
                    r=vals_norm + [vals_norm[0]],
                    theta=features_comp + [features_comp[0]],
                    fill='toself',
                    name=row.get('apelido', f'Jogador {i+1}'),
                    line_color=cores[i % len(cores)],
                    opacity=0.7,
                ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=True,
                title="Radar de Comparação",
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.info("Instale plotly para ver o gráfico radar: `pip install plotly`")

        st.dataframe(df_comp[features_comp].T.rename(columns=dict(zip(df_comp.index, df_comp['apelido'].tolist()))), use_container_width=True)
    else:
        st.info("Selecione pelo menos 2 jogadores para comparar.")

with tab_custo:
    st.subheader("💡 Análise de Custo-Benefício")
    try:
        import plotly.express as px
        fig2 = px.scatter(
            df_filtered,
            x='preco',
            y='mega_score',
            color='pos_nome',
            size='media',
            hover_name='apelido' if 'apelido' in df_filtered.columns else None,
            hover_data=['media', 'pontos_ultimas_5', 'jogos'],
            title="Mega Score vs Preço (tamanho = Média histórica)",
            labels={'preco': 'Preço (C$)', 'mega_score': 'Mega Score', 'pos_nome': 'Posição'},
            template='plotly_dark',
        )
        st.plotly_chart(fig2, use_container_width=True)
    except ImportError:
        st.info("Instale plotly: `pip install plotly`")

    st.subheader("🏅 Top Custo-Benefício por Posição")
    for pos in pos_filter:
        df_pos = df_filtered[df_filtered['pos_nome'] == pos].head(5)
        if not df_pos.empty:
            with st.expander(f"{pos} — Top 5 Custo-Benefício"):
                cols_cb = [c for c in ['apelido', 'mega_score', 'custo_beneficio', 'media', 'preco', 'status'] if c in df_pos.columns]
                st.dataframe(df_pos[cols_cb], use_container_width=True, hide_index=True)
