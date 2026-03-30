#!/usr/bin/env python3
"""
pages/02_backtesting.py

Página Streamlit — Backtesting Automático
Simula ganhos do time otimizado em rodadas históricas.
"""

import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).parent.parent
import sys; sys.path.insert(0, str(ROOT_DIR))

from src.analysis.backtesting import run_backtesting, summarize_backtesting

# ─────────────────────────────────────────────────────────────
# CONFIG DA PÁGINA
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Backtesting · Cartola FC",
    page_icon="📈",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.hero-header {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 24px;
    border: 1px solid rgba(88,166,255,0.2);
    box-shadow: 0 8px 24px rgba(0,0,0,0.3);
}
.hero-header h1 { color: #58a6ff; font-size: 2rem; font-weight: 800; margin: 0; letter-spacing: -0.5px; }
.hero-header p  { color: rgba(255,255,255,0.75); margin: 6px 0 0; font-size: 1rem; }
.metric-card {
    background: var(--secondary-background-color);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 18px 22px;
    text-align: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}
.metric-val { font-size: 2rem; font-weight: 800; color: #58a6ff; }
.metric-lbl { font-size: 0.78rem; color: #8b949e; text-transform: uppercase; letter-spacing: .06em; margin-top: 4px; }
.positive { color: #39d353 !important; }
.negative { color: #ff7b72 !important; }
.neutral  { color: #ffa657 !important; }
hr.custom { border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 28px 0; }
.alert-info { background:#1a2a3a; border:1px solid #58a6ff; border-radius:8px; padding:12px 16px; color:#cae8ff; }
.alert-warn { background:#3a2e1a; border:1px solid #ffa657; border-radius:8px; padding:12px 16px; color:#ffd8b0; }
.section-title { font-size: 1.1rem; font-weight: 700; color: #e6edf3; margin-bottom: 12px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <h1>📈 Backtesting Automático</h1>
    <p>Simula o desempenho do seu time otimizado em rodadas históricas e compara contra a média do mercado.</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CONFIGURAÇÃO DO BANCO
# ─────────────────────────────────────────────────────────────
try:
    import yaml
    with open(ROOT_DIR / "config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    db_path = ROOT_DIR / cfg.get("database", {}).get("path", "data/cartola.db")
    ano_temporada = int(cfg.get("season", {}).get("year", 2024))
except Exception:
    db_path = ROOT_DIR / "data" / "cartola.db"
    ano_temporada = 2024

if not db_path.exists():
    st.markdown('<div class="alert-warn">⚠️ Banco de dados não encontrado. Execute o pipeline principal primeiro (app.py).</div>', unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────────────────────
# PAINEL DE CONTROLE
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configurações do Backtest")
    formacao_bt = st.selectbox(
        "Formação simulada",
        ["4-3-3", "4-4-2", "3-5-2", "3-4-3", "4-5-1", "5-3-2", "5-4-1"],
        index=0,
    )
    patrimonio_bt = st.number_input(
        "Patrimônio (C$)", min_value=50.0, max_value=200.0, value=100.0, step=5.0
    )
    n_rodadas_bt = st.slider(
        "Rodadas a testar", min_value=3, max_value=38, value=10, step=1
    )
    janela_ewm_bt = st.slider(
        "Janela EWM (rodadas anteriores p/ estimar)", min_value=2, max_value=10, value=5
    )
    usar_salvas = st.toggle(
        "Usar escalações salvas quando disponíveis", value=True
    )
    rodar_bt = st.button("▶ Rodar Backtesting", type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────────────
# EXECUÇÃO
# ─────────────────────────────────────────────────────────────
if rodar_bt:
    with st.spinner("Simulando rodadas históricas..."):
        bt_df = run_backtesting(
            db_path=db_path,
            ano=ano_temporada,
            formacao=formacao_bt,
            patrimonio=patrimonio_bt,
            n_rodadas=n_rodadas_bt,
            janela_ewm=janela_ewm_bt,
            usar_escalacoes_salvas=usar_salvas,
        )
    st.session_state["bt_df"] = bt_df

bt_df: pd.DataFrame = st.session_state.get("bt_df", pd.DataFrame())

# ─────────────────────────────────────────────────────────────
# RESULTADOS
# ─────────────────────────────────────────────────────────────
if bt_df.empty:
    st.markdown(
        '<div class="alert-info">ℹ️ Configure os parâmetros na barra lateral e clique <b>Rodar Backtesting</b>.</div>',
        unsafe_allow_html=True,
    )
else:
    summary = summarize_backtesting(bt_df)

    # ── KPIs ──────────────────────────────────────────────────
    st.markdown("### 🏆 Resumo das Simulações")
    c1, c2, c3, c4, c5 = st.columns(5)

    def _kpi(col, valor, label, classe=""):
        col.markdown(
            f'<div class="metric-card"><div class="metric-val {classe}">{valor}</div>'
            f'<div class="metric-lbl">{label}</div></div>',
            unsafe_allow_html=True,
        )

    _kpi(c1, summary.get("total_rodadas", 0), "Rodadas Simuladas")
    _kpi(c2, f"{summary.get('media_pontos', 0):.1f}", "Média Pts/Rodada", "positive")
    ganho = summary.get("ganho_total_vs_mercado", 0)
    _kpi(c3, f"{ganho:+.1f}", "Ganho vs Mercado", "positive" if ganho >= 0 else "negative")
    _kpi(c4, f"{summary.get('win_rate_pct', 0):.0f}%", "Rodadas Acima da Média", "positive" if summary.get('win_rate_pct', 0) >= 50 else "neutral")
    _kpi(c5, f"{summary.get('media_cobertura', 0):.0f}%", "Cobertura Média Atletas")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Gráfico de evolução ───────────────────────────────────
    st.markdown("<hr class='custom'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📊 Pontuação Real vs Média do Mercado (por rodada)</div>", unsafe_allow_html=True)

    chart_df = bt_df[["rodada", "pontos_reais", "pontos_medio_time_mercado"]].set_index("rodada").sort_index()
    chart_df.columns = ["Time Otimizado (pts reais)", "Média do Mercado (pts)"]
    st.line_chart(chart_df)

    # ── Gráfico de ganho vs mercado ───────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>⚡ Ganho/Perda vs Média do Mercado por Rodada</div>", unsafe_allow_html=True)
    ganho_df = bt_df[["rodada", "ganho_vs_mercado"]].set_index("rodada").sort_index()
    st.bar_chart(ganho_df)

    # ── Tabela detalhada ─────────────────────────────────────
    st.markdown("<hr class='custom'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📋 Detalhamento por Rodada</div>", unsafe_allow_html=True)

    display_bt = bt_df.copy().sort_values("rodada", ascending=False).rename(
        columns={
            "rodada": "Rodada",
            "pontos_reais": "Pts Reais",
            "pontos_media_mercado": "Média Mercado",
            "pontos_medio_time_mercado": "Time c/ Média",
            "patrimonio_gasto": "C$ Gasto",
            "n_atletas": "Atletas c/ Dados",
            "cobertura_pct": "Cobertura %",
            "ganho_vs_mercado": "Ganho vs Média",
            "estrategia": "Estratégia",
            "formacao": "Formação",
        }
    )

    def _style_ganho(val):
        try:
            v = float(val)
            if v > 0:
                return "color: #39d353; font-weight: 700"
            elif v < 0:
                return "color: #ff7b72; font-weight: 700"
        except Exception:
            pass
        return ""

    st.dataframe(
        display_bt.style.applymap(_style_ganho, subset=["Ganho vs Média"]),
        use_container_width=True,
        hide_index=True,
    )

    # ── Download ──────────────────────────────────────────────
    st.markdown("<hr class='custom'>", unsafe_allow_html=True)
    csv_bt = bt_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Baixar resultado do backtest (.csv)",
        csv_bt,
        file_name=f"backtesting_{ano_temporada}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # ── Insights automáticos ──────────────────────────────────
    st.markdown("<hr class='custom'>", unsafe_allow_html=True)
    st.markdown("### 💡 Insights Automáticos")

    insights = []
    if summary.get("win_rate_pct", 0) >= 60:
        insights.append("✅ **Estratégia consistente:** o modelo superou a média do mercado em mais de 60% das rodadas simuladas.")
    elif summary.get("win_rate_pct", 0) >= 40:
        insights.append("⚠️ **Estratégia mediana:** resultados acima da média em cerca de metade das rodadas. Tente ajustar a formação ou aumentar a janela EWM.")
    else:
        insights.append("❌ **Estratégia abaixo da média:** menos de 40% das rodadas superaram o mercado. Revise a formação e os parâmetros genéticos.")

    melhor_rodada = bt_df.loc[bt_df["pontos_reais"].idxmax()]
    pior_rodada = bt_df.loc[bt_df["pontos_reais"].idxmin()]
    insights.append(f"🏆 **Melhor rodada:** #{int(melhor_rodada['rodada'])} com **{melhor_rodada['pontos_reais']:.1f} pts** reais.")
    insights.append(f"📉 **Pior rodada:** #{int(pior_rodada['rodada'])} com **{pior_rodada['pontos_reais']:.1f} pts** reais.")

    if summary.get("ganho_total_vs_mercado", 0) > 0:
        insights.append(f"💰 **Saldo positivo:** o modelo acumulou **+{summary['ganho_total_vs_mercado']:.1f} pontos** acima da média de mercado no período.")
    else:
        insights.append(f"📊 **Saldo negativo:** o modelo ficou **{summary['ganho_total_vs_mercado']:.1f} pontos** abaixo da média de mercado no período. Considere aumentar as gerações do AG.")

    for ins in insights:
        st.markdown(f"> {ins}")
