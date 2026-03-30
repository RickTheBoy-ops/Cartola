"""
Página Streamlit: Backtesting Automático.
Visualização interativa dos resultados do backtesting por rodada.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtesting import executar_backtesting, calcular_metricas_gerais

st.set_page_config(
    page_title="Backtesting | Cartola IA",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Backtesting Automático")
st.caption("Simulação das escalações da IA em rodadas históricas vs. melhor escalação possível")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Parâmetros")
    budget = st.number_input("Orçamento (Cartoletas)", min_value=50.0, max_value=200.0, value=100.0, step=5.0)
    formacao = st.selectbox("Formação", ["4-3-3", "4-4-2", "3-5-2", "4-5-1"], index=0)
    executar = st.button("▶️ Executar Backtesting", type="primary")


# ---------------------------------------------------------------------------
# Execução
# ---------------------------------------------------------------------------

if executar or "bt_resultado" not in st.session_state:
    with st.spinner("Executando backtesting em todas as rodadas disponíveis..."):
        st.session_state["bt_resultado"] = executar_backtesting(
            budget=budget, formacao=formacao, salvar_csv=True
        )
        st.session_state["bt_metricas"] = calcular_metricas_gerais(st.session_state["bt_resultado"])

df_bt: pd.DataFrame = st.session_state["bt_resultado"]
metricas: dict = st.session_state["bt_metricas"]

if df_bt.empty:
    st.warning("Nenhum dado histórico encontrado. Coloque arquivos rodada_N.csv ou rodada_N.json em data/")
    st.stop()


# ---------------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------------

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("🏆 Rodadas Analisadas", metricas.get("total_rodadas", 0))
col2.metric("📈 Média Pontos IA", metricas.get("media_pontos_ia", 0))
col3.metric("🔮 Média Melhor Possível", metricas.get("media_pontos_melhor", 0))
col4.metric("✅ % Superou Média Geral", f"{metricas.get('pct_superou_media', 0):.1f}%")
col5.metric("📉 RMSE Médio", metricas.get("rmse_medio", 0))

st.divider()


# ---------------------------------------------------------------------------
# Gráfico de evolução
# ---------------------------------------------------------------------------

st.subheader("📈 Evolução de Pontuação por Rodada")

fig_evo = go.Figure()
fig_evo.add_trace(go.Scatter(
    x=df_bt["rodada"], y=df_bt["pontos_ia"],
    mode="lines+markers", name="IA",
    line=dict(color="#00D4FF", width=2),
    marker=dict(size=7),
))
fig_evo.add_trace(go.Scatter(
    x=df_bt["rodada"], y=df_bt["pontos_melhor_possivel"],
    mode="lines+markers", name="Melhor Possível",
    line=dict(color="#FFD700", width=2, dash="dash"),
    marker=dict(size=7),
))
fig_evo.add_trace(go.Scatter(
    x=df_bt["rodada"], y=df_bt["pontos_media_geral"],
    mode="lines", name="Média Geral",
    line=dict(color="#FF6B6B", width=1.5, dash="dot"),
))
fig_evo.update_layout(
    xaxis_title="Rodada",
    yaxis_title="Pontuação",
    template="plotly_dark",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=400,
)
st.plotly_chart(fig_evo, use_container_width=True)


# ---------------------------------------------------------------------------
# Delta vs Melhor Possível
# ---------------------------------------------------------------------------

st.subheader("📉 Gap IA vs Melhor Escalação Possível")
fig_delta = px.bar(
    df_bt, x="rodada", y="delta_vs_melhor",
    color="delta_vs_melhor",
    color_continuous_scale="RdYlGn_r",
    labels={"delta_vs_melhor": "Pontos Abaixo do Melhor", "rodada": "Rodada"},
    title="Diferença entre IA e Melhor Escalação Possível (Hindsight)",
    template="plotly_dark",
)
fig_delta.update_layout(height=350)
st.plotly_chart(fig_delta, use_container_width=True)


# ---------------------------------------------------------------------------
# MAE / RMSE por rodada
# ---------------------------------------------------------------------------

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("📊 Erro de Previsão (MAE) por Rodada")
    fig_mae = px.line(
        df_bt, x="rodada", y="mae",
        markers=True,
        labels={"mae": "MAE", "rodada": "Rodada"},
        template="plotly_dark",
        color_discrete_sequence=["#FF9F40"],
    )
    st.plotly_chart(fig_mae, use_container_width=True)

with col_b:
    st.subheader("📊 % Superou Média Geral")
    total = len(df_bt)
    superou = df_bt["superou_media"].sum()
    nao_superou = total - superou
    fig_pizza = px.pie(
        names=["Superou", "Não Superou"],
        values=[superou, nao_superou],
        color_discrete_map={"Superou": "#2ecc71", "Não Superou": "#e74c3c"},
        template="plotly_dark",
    )
    st.plotly_chart(fig_pizza, use_container_width=True)


# ---------------------------------------------------------------------------
# Tabela detalhada
# ---------------------------------------------------------------------------

st.divider()
st.subheader("📋 Resultado por Rodada")

df_display = df_bt[["rodada", "pontos_ia", "pontos_melhor_possivel", "delta_vs_melhor",
                    "delta_vs_media", "superou_media", "mae", "rmse"]].copy()
df_display["superou_media"] = df_display["superou_media"].map({True: "✅", False: "❌"})
df_display.columns = ["Rodada", "Pontos IA", "Melhor Possível", "Gap Hindsight",
                      "Delta vs Média", "Superou Média", "MAE", "RMSE"]
st.dataframe(
    df_display.style.background_gradient(subset=["Pontos IA"], cmap="RdYlGn"),
    use_container_width=True,
    hide_index=True,
)

# Botão de download
csv = df_bt.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Baixar CSV Completo",
    data=csv,
    file_name="backtesting_results.csv",
    mime="text/csv",
)
