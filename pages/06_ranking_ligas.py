#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
========================================================================
RANKING DE LIGAS
========================================================================
Classificação e comparação entre times de uma liga.
Inspirада na funcionalidade 'Parciais por Ligas' da Máquina do Cartola.

Funcionalidades:
  - Tabela de classificação da liga
  - Gráfico de evolução de pontos por rodada
  - Comparativo entre times da liga
  - Histórico de pontuações
  - Análise de tendência (times em alta/baixa)
========================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="Ranking de Ligas | Cartola AI",
    page_icon="🏆",
    layout="wide",
)

st.title("🏆 Ranking de Ligas")
st.markdown("Classificação e análise de desempenho dos times da sua liga.")


@st.cache_data(ttl=300)
def gerar_dados_liga(n_times=10, n_rodadas=15, seed=42):
    """Gera dados demo de liga. Em produção, conectar API Cartola."""
    import glob
    csvs = glob.glob("data/liga*.csv")
    if csvs:
        return pd.read_csv(csvs[0])

    rng = np.random.default_rng(seed)
    times = [f"Time_{chr(65+i)}" for i in range(n_times)]
    registros = []
    for time in times:
        base = rng.uniform(40, 90)
        for rodada in range(1, n_rodadas + 1):
            pontos = max(0, round(base + rng.normal(0, 12), 2))
            registros.append({'time': time, 'rodada': rodada, 'pontos': pontos})
    return pd.DataFrame(registros)


df_liga = gerar_dados_liga()

# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configurações")
    liga_nome = st.text_input("Nome da Liga", "Minha Liga")
    times_disponiveis = df_liga['time'].unique().tolist()
    times_selecionados = st.multiselect("Times na liga", times_disponiveis, default=times_disponiveis)
    rodada_atual = st.slider("Rodadas consideradas", 1, int(df_liga['rodada'].max()), int(df_liga['rodada'].max()))

# ── Filtrar ───────────────────────────────────────────────────────────
df_filtrado = df_liga[
    (df_liga['time'].isin(times_selecionados)) &
    (df_liga['rodada'] <= rodada_atual)
]

# ── Classificação ─────────────────────────────────────────────────────
df_class = df_filtrado.groupby('time').agg(
    total_pontos=('pontos', 'sum'),
    media_pontos=('pontos', 'mean'),
    melhor_rodada=('pontos', 'max'),
    pior_rodada=('pontos', 'min'),
    rodadas_jogadas=('pontos', 'count'),
).reset_index().sort_values('total_pontos', ascending=False).reset_index(drop=True)
df_class.index += 1
df_class['posicao'] = df_class.index
df_class['tendencia'] = df_class['time'].apply(
    lambda t: '📈' if df_filtrado[df_filtrado['time'] == t].tail(3)['pontos'].mean() >
    df_filtrado[df_filtrado['time'] == t].head(3)['pontos'].mean() else '📉'
)

st.subheader(f"📋 Classificação — {liga_nome} (até rodada {rodada_atual})")

# Top 3
cols_top = st.columns(3)
medal = ["🥇", "🥈", "🥉"]
for i, col in enumerate(cols_top):
    if i < len(df_class):
        row = df_class.iloc[i]
        col.markdown(f"""
        <div style='background:linear-gradient(135deg,#1a2a4a,#2a4a6a);border-radius:12px;
        padding:16px;text-align:center;color:white;border:1px solid #3a6a9a;'>
            <div style='font-size:28px'>{medal[i]}</div>
            <div style='font-size:16px;font-weight:700'>{row['time']}</div>
            <div style='font-size:22px;color:#FFD700;font-weight:700'>{row['total_pontos']:.1f} pts</div>
            <div style='color:#aac;font-size:12px'>Média: {row['media_pontos']:.1f} | {row['tendencia']}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("")
st.dataframe(
    df_class[['posicao', 'time', 'total_pontos', 'media_pontos', 'melhor_rodada', 'pior_rodada', 'rodadas_jogadas', 'tendencia']],
    use_container_width=True,
    hide_index=True,
)

st.divider()

# ── Gráfico de evolução ───────────────────────────────────────────────
tab_evolucao, tab_comparar, tab_tendencia = st.tabs(["📈 Evolução", "⚖️ Comparar Times", "🔥 Tendência"])

with tab_evolucao:
    st.subheader("📈 Evolução de Pontos por Rodada")
    try:
        import plotly.express as px
        df_evo = df_filtrado.sort_values('rodada')
        df_evo['pontos_acum'] = df_evo.groupby('time')['pontos'].cumsum()
        fig = px.line(
            df_evo, x='rodada', y='pontos_acum', color='time',
            title='Pontuação Acumulada por Rodada',
            labels={'rodada': 'Rodada', 'pontos_acum': 'Pontos Acumulados', 'time': 'Time'},
            template='plotly_dark',
            markers=True,
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.info("Instale plotly: `pip install plotly`")

with tab_comparar:
    st.subheader("⚖️ Comparar Times")
    times_comp = st.multiselect("Selecione times", times_selecionados, default=times_selecionados[:3])
    if times_comp:
        df_comp = df_filtrado[df_filtrado['time'].isin(times_comp)]
        try:
            import plotly.express as px
            fig2 = px.bar(
                df_comp, x='rodada', y='pontos', color='time',
                barmode='group', title='Pontos por Rodada — Comparativo',
                template='plotly_dark',
            )
            st.plotly_chart(fig2, use_container_width=True)
        except ImportError:
            st.table(df_comp.pivot(index='rodada', columns='time', values='pontos'))

with tab_tendencia:
    st.subheader("🔥 Times em Alta e em Baixa")
    tendencias = []
    for time in times_selecionados:
        df_t = df_filtrado[df_filtrado['time'] == time].sort_values('rodada')
        if len(df_t) >= 4:
            ultimas = df_t.tail(3)['pontos'].mean()
            anteriores = df_t.iloc[-6:-3]['pontos'].mean() if len(df_t) >= 6 else df_t.head(3)['pontos'].mean()
            delta = ultimas - anteriores
            tendencias.append({'time': time, 'media_ultimas_3': round(ultimas, 2),
                               'media_anteriores': round(anteriores, 2),
                               'delta': round(delta, 2),
                               'tendencia': '📈 Alta' if delta > 2 else ('📉 Baixa' if delta < -2 else '➡️ Estável')})

    if tendencias:
        df_tend = pd.DataFrame(tendencias).sort_values('delta', ascending=False)
        col_alta, col_baixa = st.columns(2)
        with col_alta:
            st.markdown("**🔥 Times em Alta**")
            st.dataframe(df_tend[df_tend['delta'] > 0].head(5), use_container_width=True, hide_index=True)
        with col_baixa:
            st.markdown("**❄️ Times em Baixa**")
            st.dataframe(df_tend[df_tend['delta'] <= 0].head(5), use_container_width=True, hide_index=True)
