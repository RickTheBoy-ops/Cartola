"""
Dashboard Interativo - Visualização de Time e Táticas.
Página Streamlit para exibir o time escalado em formato de campo,
gráficos de radar, mapas de calor e alertas de notícias.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Garante que src/ está no path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.news_agent import analisar_noticias_jogadores

# ---------------------------------------------------------------------------
# Configuração da página
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Time & Táticas | Cartola IA",
    page_icon="⚽",
    layout="wide",
)

st.title("⚽ Visualização de Time e Táticas")
st.caption("Dashboard interativo para análise do seu time no Cartola FC")


# ---------------------------------------------------------------------------
# Dados de exemplo (substituir por integração real com a API do Cartola)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def _carregar_time_exemplo() -> pd.DataFrame:
    """Retorna dados de exemplo do time escalado."""
    return pd.DataFrame([
        {"nome": "Everson",     "posicao": "GOL", "clube": "Atlético-MG", "preco": 6.50,  "media": 6.2, "ultima_pontuacao": 7.0,  "valorizacao": 0.5,  "foto_url": ""},
        {"nome": "Guilherme",   "posicao": "LAT", "clube": "Palmeiras",   "preco": 9.20,  "media": 5.8, "ultima_pontuacao": 4.5,  "valorizacao": -0.3, "foto_url": ""},
        {"nome": "Murillo",     "posicao": "ZAG", "clube": "Corinthians",  "preco": 8.10,  "media": 6.1, "ultima_pontuacao": 8.2,  "valorizacao": 1.2,  "foto_url": ""},
        {"nome": "Léo Ortiz",   "posicao": "ZAG", "clube": "Flamengo",    "preco": 10.50, "media": 6.5, "ultima_pontuacao": 6.0,  "valorizacao": 0.2,  "foto_url": ""},
        {"nome": "Ayrton Lucas","posicao": "LAT", "clube": "Flamengo",    "preco": 11.30, "media": 7.0, "ultima_pontuacao": 9.5,  "valorizacao": 2.1,  "foto_url": ""},
        {"nome": "Gerson",      "posicao": "MEI", "clube": "Flamengo",    "preco": 15.70, "media": 7.5, "ultima_pontuacao": 11.2, "valorizacao": 3.5,  "foto_url": ""},
        {"nome": "Veiga",       "posicao": "MEI", "clube": "Palmeiras",   "preco": 18.40, "media": 8.2, "ultima_pontuacao": 7.0,  "valorizacao": 0.0,  "foto_url": ""},
        {"nome": "De La Cruz",  "posicao": "MEI", "clube": "Flamengo",    "preco": 16.20, "media": 7.8, "ultima_pontuacao": 8.5,  "valorizacao": 1.8,  "foto_url": ""},
        {"nome": "Luiz Henrique","posicao": "ATA", "clube": "Flamengo",   "preco": 14.50, "media": 7.2, "ultima_pontuacao": 5.5,  "valorizacao": -0.5, "foto_url": ""},
        {"nome": "Pedro",       "posicao": "ATA", "clube": "Flamengo",    "preco": 20.10, "media": 8.8, "ultima_pontuacao": 12.5, "valorizacao": 4.2,  "foto_url": ""},
        {"nome": "Raphinha",    "posicao": "ATA", "clube": "Barcelona",   "preco": 22.30, "media": 9.1, "ultima_pontuacao": 10.0, "valorizacao": 2.8,  "foto_url": ""},
        {"nome": "Abel Ferreira","posicao": "TEC", "clube": "Palmeiras",  "preco": 5.00,  "media": 6.0, "ultima_pontuacao": 6.5,  "valorizacao": 0.5,  "foto_url": ""},
    ])


def _posicoes_campo(formacao: str) -> dict[str, list[tuple[float, float]]]:
    """Retorna coordenadas (x, y) por posição para visualização no campo.
    O campo tem x em [0,1] (horizontal) e y em [0,1] (vertical, 0=defesa).
    """
    layouts = {
        "4-3-3": {
            "GOL": [(0.5, 0.05)],
            "LAT": [(0.1, 0.22), (0.9, 0.22)],
            "ZAG": [(0.35, 0.22), (0.65, 0.22)],
            "MEI": [(0.2, 0.50), (0.5, 0.50), (0.8, 0.50)],
            "ATA": [(0.2, 0.78), (0.5, 0.78), (0.8, 0.78)],
            "TEC": [(0.5, 0.95)],
        },
        "4-4-2": {
            "GOL": [(0.5, 0.05)],
            "LAT": [(0.1, 0.22), (0.9, 0.22)],
            "ZAG": [(0.35, 0.22), (0.65, 0.22)],
            "MEI": [(0.15, 0.55), (0.38, 0.55), (0.62, 0.55), (0.85, 0.55)],
            "ATA": [(0.35, 0.82), (0.65, 0.82)],
            "TEC": [(0.5, 0.95)],
        },
        "3-5-2": {
            "GOL": [(0.5, 0.05)],
            "ZAG": [(0.25, 0.20), (0.5, 0.20), (0.75, 0.20)],
            "LAT": [],
            "MEI": [(0.1, 0.50), (0.3, 0.50), (0.5, 0.50), (0.7, 0.50), (0.9, 0.50)],
            "ATA": [(0.35, 0.82), (0.65, 0.82)],
            "TEC": [(0.5, 0.95)],
        },
    }
    return layouts.get(formacao, layouts["4-3-3"])


def _desenhar_campo(df_time: pd.DataFrame, formacao: str) -> go.Figure:
    """Desenha o campo de futebol com os jogadores posicionados."""
    fig = go.Figure()

    # Fundo do campo (verde)
    fig.add_shape(type="rect", x0=0, y0=0, x1=1, y1=1,
                  fillcolor="#2d6a2d", line=dict(color="white", width=2))
    # Linha do meio
    fig.add_shape(type="line", x0=0, y0=0.5, x1=1, y1=0.5,
                  line=dict(color="white", width=1.5))
    # Círculo central
    fig.add_shape(type="circle", x0=0.38, y0=0.40, x1=0.62, y1=0.60,
                  line=dict(color="white", width=1.5))
    # Área defensiva
    fig.add_shape(type="rect", x0=0.25, y0=0.0, x1=0.75, y1=0.15,
                  line=dict(color="white", width=1.5))
    # Área ofensiva
    fig.add_shape(type="rect", x0=0.25, y0=0.85, x1=0.75, y1=1.0,
                  line=dict(color="white", width=1.5))

    coord_map = _posicoes_campo(formacao)
    posicao_idx: dict[str, int] = {}

    # Cores por posição
    cores = {"GOL": "#FFD700", "ZAG": "#4169E1", "LAT": "#1E90FF",
             "MEI": "#32CD32", "ATA": "#FF4500", "TEC": "#9400D3"}

    for _, jogador in df_time.iterrows():
        pos = jogador["posicao"]
        coords = coord_map.get(pos, [])
        idx = posicao_idx.get(pos, 0)

        if idx >= len(coords):
            continue

        x, y = coords[idx]
        posicao_idx[pos] = idx + 1
        cor = cores.get(pos, "white")

        # Círculo do jogador
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode="markers+text",
            marker=dict(size=28, color=cor, line=dict(color="white", width=2)),
            text=[jogador["nome"].split()[0]],
            textposition="bottom center",
            textfont=dict(color="white", size=10),
            hovertemplate=(
                f"<b>{jogador['nome']}</b><br>"
                f"Clube: {jogador['clube']}<br>"
                f"Média: {jogador['media']}<br>"
                f"Última rodada: {jogador['ultima_pontuacao']}<br>"
                f"Preço: C$ {jogador['preco']:.1f}<br>"
                f"Valorização: {jogador['valorizacao']:+.1f}"
                "<extra></extra>"
            ),
            showlegend=False,
        ))

    fig.update_layout(
        xaxis=dict(range=[-0.05, 1.05], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-0.1, 1.1], showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="#2d6a2d",
        paper_bgcolor="#1e1e1e",
        height=550,
        margin=dict(l=10, r=10, t=30, b=10),
        title=dict(text=f"Formação: {formacao}", font=dict(color="white", size=14), x=0.5),
    )
    return fig


def _radar_chart(df_pos: pd.DataFrame, jogadores_sel: list[str]) -> go.Figure:
    """Gera gráfico de radar comparando atributos de jogadores."""
    categorias = ["Média", "Última Pont.", "Valoriz.", "Custo/Ponto"]

    fig = go.Figure()
    for nome in jogadores_sel:
        row = df_pos[df_pos["nome"] == nome]
        if row.empty:
            continue
        r = row.iloc[0]
        custo_ponto = r["preco"] / max(r["media"], 0.1)
        custo_ponto_norm = max(0, 10 - custo_ponto)  # inverso: menor custo = melhor
        valores = [
            float(r["media"]),
            float(r["ultima_pontuacao"]),
            float(r["valorizacao"]) + 5,  # shift para ficar positivo
            float(custo_ponto_norm),
        ]
        fig.add_trace(go.Scatterpolar(
            r=valores + [valores[0]],
            theta=categorias + [categorias[0]],
            fill="toself",
            name=nome,
            opacity=0.7,
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 12])),
        showlegend=True,
        paper_bgcolor="#1e1e1e",
        font=dict(color="white"),
        height=380,
        title="Comparativo de Atributos (Radar)",
    )
    return fig


# ---------------------------------------------------------------------------
# Layout principal
# ---------------------------------------------------------------------------

def main() -> None:
    df_time = _carregar_time_exemplo()

    # --- Sidebar: Filtros ---
    with st.sidebar:
        st.header("⚙️ Configurações")
        formacao = st.selectbox("Formação Tática", ["4-3-3", "4-4-2", "3-5-2", "4-5-1"], index=0)
        preco_max = st.slider("Preço Máximo (C$)", 4.0, 30.0, 25.0, 0.5)
        media_min = st.slider("Média Mínima", 0.0, 10.0, 4.0, 0.5)
        buscar_news = st.checkbox("🔴 Buscar Notícias em Tempo Real", value=False)

    df_filtrado = df_time[
        (df_time["preco"] <= preco_max) | (df_time["posicao"] == "TEC")
    ].copy()
    df_filtrado = df_filtrado[
        (df_filtrado["media"] >= media_min) | (df_filtrado["posicao"] == "TEC")
    ]

    # --- KPIs ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👥 Jogadores", len(df_time) - 1)
    col2.metric("💰 Custo Total", f"C$ {df_time['preco'].sum():.1f}")
    col3.metric("📊 Média Esperada", f"{df_time[df_time['posicao'] != 'TEC']['media'].mean():.1f}")
    col4.metric("📈 Pontos Última Rodada", f"{df_time[df_time['posicao'] != 'TEC']['ultima_pontuacao'].sum():.1f}")

    st.divider()

    # --- Campo de Futebol ---
    st.subheader("🏟️ Formação Tática")
    fig_campo = _desenhar_campo(df_time, formacao)
    st.plotly_chart(fig_campo, use_container_width=True)

    st.divider()

    # --- Tabela de Jogadores ---
    st.subheader("📋 Elenco Escalado")
    col_pos_filter = st.multiselect(
        "Filtrar por posição",
        options=df_time["posicao"].unique().tolist(),
        default=df_time["posicao"].unique().tolist(),
    )
    df_tabela = df_time[df_time["posicao"].isin(col_pos_filter)].copy()
    df_tabela["valorização"] = df_tabela["valorizacao"].apply(lambda x: f"{x:+.1f}")
    st.dataframe(
        df_tabela[["nome", "posicao", "clube", "preco", "media", "ultima_pontuacao", "valorização"]]
        .rename(columns={"nome": "Jogador", "posicao": "Pos", "clube": "Clube",
                         "preco": "Preço (C$)", "media": "Média",
                         "ultima_pontuacao": "Ult. Rodada"})
        .style.background_gradient(subset=["Média", "Ult. Rodada"], cmap="RdYlGn"),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # --- Gráfico de Radar ---
    st.subheader("🕸️ Comparativo Radar")
    posicoes_disp = [p for p in df_time["posicao"].unique() if p != "TEC"]
    pos_radar = st.selectbox("Posição para comparar", posicoes_disp)
    df_pos_radar = df_time[df_time["posicao"] == pos_radar]
    jogadores_radar = st.multiselect(
        "Selecione jogadores",
        df_pos_radar["nome"].tolist(),
        default=df_pos_radar["nome"].tolist()[:2],
    )
    if jogadores_radar:
        fig_radar = _radar_chart(df_pos_radar, jogadores_radar)
        st.plotly_chart(fig_radar, use_container_width=True)

    st.divider()

    # --- Mapa de Calor de Valorização ---
    st.subheader("🌡️ Valorização por Jogador")
    df_heat = df_time[df_time["posicao"] != "TEC"].sort_values("valorizacao", ascending=True)
    fig_bar = px.bar(
        df_heat, x="valorizacao", y="nome", orientation="h",
        color="valorizacao", color_continuous_scale="RdYlGn",
        labels={"valorizacao": "Valorização (C$)", "nome": ""},
        title="Valorização na Última Rodada",
        template="plotly_dark",
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # --- Alertas de Notícias ---
    st.divider()
    st.subheader("📰 Alertas de Notícias")

    if buscar_news:
        jogadores_busca = df_time[df_time["posicao"] != "TEC"]["nome"].tolist()
        with st.spinner("Buscando notícias recentes..."):
            noticias = analisar_noticias_jogadores(jogadores_busca, horas=24)

        for jogador, info in noticias.items():
            if info["total_noticias"] == 0:
                continue
            icone = "🟢" if info["impacto"] == "positivo" else ("🔴" if info["impacto"] == "negativo" else "🟡")
            with st.expander(f"{icone} {jogador} — Score: {info['score']:+.2f} ({info['total_noticias']} notícias)"):
                for n in info["noticias"][:3]:
                    st.markdown(f"**{n['titulo']}**")
                    st.caption(f"{n['fonte']} • {n['data'][:10]}")
                    st.write(n['resumo'])
                    if n['url']:
                        st.markdown(f"[Leia mais]({n['url']})")
                    st.divider()
    else:
        st.info("Ative 'Buscar Notícias em Tempo Real' no menu lateral para ver alertas.")


if __name__ == "__main__":
    main()
