#!/usr/bin/env python3
"""
pages/01_taticas.py

Página Streamlit — Dashboard Tático Interativo
Visualiza a escalação atual com formação, gráfico de campo e estatísticas.
"""

import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).parent.parent
import sys; sys.path.insert(0, str(ROOT_DIR))

# ─────────────────────────────────────────────────────────────
# CONFIG DA PÁGINA
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Táticas · Cartola FC",
    page_icon="⚽",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.hero-header {
    background: linear-gradient(135deg, #006b33 0%, #00a651 50%, #00d669 100%);
    border-radius: 16px; padding: 28px 36px; margin-bottom: 24px;
    border: 1px solid #005a2b;
    box-shadow: 0 8px 24px rgba(0,166,81,0.2);
}
.hero-header h1 { color: white; font-size: 2rem; font-weight: 800; margin: 0; }
.hero-header p  { color: rgba(255,255,255,0.85); margin: 6px 0 0; font-size: 1rem; }
.field-wrapper {
    position: relative;
    background: linear-gradient(to bottom, #1a6b33 0%, #2d8b4a 50%, #1a6b33 100%);
    border-radius: 12px;
    overflow: hidden;
    border: 3px solid rgba(255,255,255,0.3);
    padding: 20px 10px;
    min-height: 520px;
}
.field-lines {
    position: absolute; inset: 0;
    background:
        linear-gradient(to bottom,
            rgba(255,255,255,0.08) 0%, rgba(255,255,255,0.08) 1px, transparent 1px, transparent 49%,
            rgba(255,255,255,0.3) 49%, rgba(255,255,255,0.3) 51%,
            transparent 51%, transparent 99%,
            rgba(255,255,255,0.08) 99%);
    pointer-events: none;
}
.player-node {
    display: flex; flex-direction: column;
    align-items: center; text-align: center;
    gap: 4px;
}
.player-avatar {
    width: 52px; height: 52px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.3rem; font-weight: 800;
    border: 3px solid rgba(255,255,255,0.5);
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
    cursor: default;
}
.player-name {
    font-size: 0.72rem; font-weight: 700; color: white;
    text-shadow: 0 1px 4px rgba(0,0,0,0.8);
    max-width: 70px; overflow: hidden;
    white-space: nowrap; text-overflow: ellipsis;
    background: rgba(0,0,0,0.45); border-radius: 4px; padding: 1px 4px;
}
.player-pts {
    font-size: 0.68rem; font-weight: 600;
    color: #ffd700; text-shadow: 0 1px 3px rgba(0,0,0,0.8);
}
.avatar-gol { background: #1a4a33; border-color: #39d353; }
.avatar-lat { background: #1a3a6b; border-color: #79c0ff; }
.avatar-zag { background: #3d4a1a; border-color: #a2e885; }
.avatar-mei { background: #4a2e1a; border-color: #ffa657; }
.avatar-ata { background: #5a1a1a; border-color: #ff7b72; }
.avatar-tec { background: #3a1a5a; border-color: #d2a8ff; }
.capitao-ring { box-shadow: 0 0 0 3px #ffd700, 0 4px 12px rgba(0,0,0,0.4) !important; }
hr.custom { border: none; border-top: 1px solid rgba(255,255,255,0.08); margin: 24px 0; }
.stat-card {
    background: var(--secondary-background-color);
    border: 1px solid rgba(255,255,255,0.08); border-radius: 10px;
    padding: 14px 16px; text-align: center;
}
.stat-val { font-size: 1.5rem; font-weight: 800; color: #00a651; }
.stat-lbl { font-size: 0.72rem; color: #8b949e; text-transform: uppercase; letter-spacing:.05em; margin-top:3px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────
POS_NOME  = {1: "GOL", 2: "LAT", 3: "ZAG", 4: "MEI", 5: "ATA", 6: "TEC"}
POS_EMOJI = {1: "🧤", 2: "🏃", 3: "🛡️", 4: "⚙️", 5: "🔥", 6: "📋"}
POS_CLS   = {1: "avatar-gol", 2: "avatar-lat", 3: "avatar-zag", 4: "avatar-mei", 5: "avatar-ata", 6: "avatar-tec"}

# Posições no campo por formação [GOL, ZAG×, LAT×, MEI×, ATA×, TEC]
FORMACOES_LINHAS = {
    "4-3-3": [[5], [3, 3], [2, 2], [4, 4, 4], [5, 5, 5], [6]],
    "4-4-2": [[5], [3, 3], [2, 2], [4, 4, 4, 4], [5, 5], [6]],
    "3-5-2": [[5], [3, 3, 3], [2], [4, 4, 4, 4, 4], [5, 5], [6]],
    "3-4-3": [[5], [3, 3, 3], [2], [4, 4, 4, 4], [5, 5, 5], [6]],
    "4-5-1": [[5], [3, 3], [2, 2], [4, 4, 4, 4, 4], [5], [6]],
    "5-3-2": [[5], [3, 3, 3], [2, 2], [4, 4, 4], [5, 5], [6]],
    "5-4-1": [[5], [3, 3, 3], [2, 2], [4, 4, 4, 4], [5], [6]],
}

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <h1>⚽ Dashboard Tático</h1>
    <p>Visualize a escalação do seu time otimizado no campo e analise o desempenho por posição.</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# DADOS DO SESSION STATE ou BANCO
# ─────────────────────────────────────────────────────────────
best_team = st.session_state.get("best_team", [])
stats     = st.session_state.get("stats", {})
formacao  = st.session_state.get("formacao", "4-3-3")
rodada    = st.session_state.get("rodada", "—")

# Se não rodou o pipeline, tenta carregar o CSV mais recente
if not best_team:
    try:
        import yaml
        with open(ROOT_DIR / "config.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        csv_path = ROOT_DIR / "data" / "processed" / "time_sugerido.csv"
        if csv_path.exists():
            df_csv = pd.read_csv(csv_path)
            best_team = df_csv.to_dict("records")
            st.info("📂 Exibindo último time otimizado salvo em disco.")
    except Exception:
        pass

if not best_team:
    st.markdown(
        '<div style="background:#1a2a3a; border:1px solid #58a6ff; border-radius:8px; padding:14px; color:#cae8ff;">'
        'ℹ️ Nenhum time otimizado disponível. Execute o pipeline principal no <b>app.py</b> primeiro.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

# ─────────────────────────────────────────────────────────────
# KPIs
# ─────────────────────────────────────────────────────────────
if stats:
    c1, c2, c3, c4 = st.columns(4)
    def _kpi_card(col, val, lbl):
        col.markdown(f'<div class="stat-card"><div class="stat-val">{val}</div><div class="stat-lbl">{lbl}</div></div>', unsafe_allow_html=True)
    _kpi_card(c1, f"{stats.get('total_pontos_preditos', 0):.1f}", "Pts Preditos")
    _kpi_card(c2, f"C$ {stats.get('total_preco', 0):.1f}", "Custo Total")
    _kpi_card(c3, formacao, "Formação")
    _kpi_card(c4, f"#{rodada}", "Rodada")
    st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# CAMPO TÁTICO
# ─────────────────────────────────────────────────────────────
st.markdown("<hr class='custom'>", unsafe_allow_html=True)
st.markdown("### 🗺️ Campo Tático")

# Organizar jogadores por posição
team_by_pos: dict = {}
for a in best_team:
    pid = int(a.get("posicao_id", 0))
    team_by_pos.setdefault(pid, []).append(a)

capitao = max(best_team, key=lambda x: float(x.get("predicao", 0))) if best_team else {}

linhas_formacao = FORMACOES_LINHAS.get(formacao, FORMACOES_LINHAS["4-3-3"])

field_html = '<div class="field-wrapper"><div class="field-lines"></div>'
field_html += f'<div style="text-align:center; color:rgba(255,255,255,0.5); font-size:0.7rem; margin-bottom:12px; letter-spacing:.1em;">GOLEIRO ↕ ATACANTES — {formacao}</div>'

# Flatten ordered player list by field position
pos_iterators = {k: iter(v) for k, v in team_by_pos.items()}

for linha in linhas_formacao:
    n = len(linha)
    field_html += f'<div style="display:flex; justify-content:space-around; align-items:center; margin: 18px 0;">'
    for pos_id in linha:
        atleta = next(pos_iterators.get(pos_id, iter([])), None)
        if atleta:
            nome  = str(atleta.get("apelido", atleta.get("nome", "—")))[:14]
            pts   = float(atleta.get("predicao", atleta.get("media", 0)))
            cls   = POS_CLS.get(pos_id, "avatar-gol")
            emoji = POS_EMOJI.get(pos_id, "⚽")
            is_cap = (atleta is capitao)
            cap_class = " capitao-ring" if is_cap else ""
            cap_label = " 👑" if is_cap else ""
            field_html += (
                f'<div class="player-node">'
                f'<div class="player-avatar {cls}{cap_class}">{emoji}</div>'
                f'<div class="player-name">{nome}{cap_label}</div>'
                f'<div class="player-pts">{pts:.1f}pts</div>'
                f'</div>'
            )
        else:
            field_html += '<div class="player-node"><div style="width:52px;height:52px;"></div></div>'
    field_html += '</div>'

field_html += '</div>'
st.markdown(field_html, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# ESTATÍSTICAS POR POSIÇÃO
# ─────────────────────────────────────────────────────────────
st.markdown("<hr class='custom'>", unsafe_allow_html=True)
st.markdown("### 📊 Pontuação por Posição")

rows_pos = []
for pid, atletas in sorted(team_by_pos.items()):
    total_pts = sum(float(a.get("predicao", 0)) for a in atletas)
    total_cost = sum(float(a.get("preco", 0)) for a in atletas)
    nomes = ", ".join(a.get("apelido", "?") for a in atletas)
    rows_pos.append({
        "Posição": POS_NOME.get(pid, "?"),
        "Atletas": nomes,
        "Pts Preditos": f"{total_pts:.1f}",
        "Custo C$": f"{total_cost:.1f}",
        "Eficiência (pts/C$)": f"{(total_pts/total_cost):.2f}" if total_cost > 0 else "—",
    })

if rows_pos:
    st.dataframe(pd.DataFrame(rows_pos), use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────
# RADAR DE EFICIÊNCIA
# ─────────────────────────────────────────────────────────────
st.markdown("<hr class='custom'>", unsafe_allow_html=True)
st.markdown("### ⚡ Distribuição de Pontos Preditos")

chart_data = {
    a.get("apelido", a.get("nome", "?")): float(a.get("predicao", 0))
    for a in sorted(best_team, key=lambda x: float(x.get("predicao", 0)), reverse=True)
}
st.bar_chart(pd.Series(chart_data).rename("Pontos Preditos"))
