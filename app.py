#!/usr/bin/env python3
"""
Cartola FC - Dashboard Streamlit
Interface visual para análise e otimização de escalação
"""

import os
import sys
import sqlite3
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import streamlit as st

# Adicionar o diretório raiz ao path
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

# ─────────────────────────────────────────────────────────────
# CONFIGURAÇÃO DA PÁGINA
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cartola FC · Optimizer",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# ESTILOS GLOBAIS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Sidebar SVG Logo Container */
.sidebar-logo {
    text-align: center;
    padding: 10px 0 20px 0;
}
.sidebar-logo svg {
    width: 64px;
    height: 64px;
    fill: #00a651;
    margin-bottom: 8px;
}
.sidebar-logo h2 {
    color: #00a651;
    font-size: 1.4rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.5px;
}
.sidebar-logo p {
    color: #8b949e;
    font-size: 0.85rem;
    margin: 0;
    font-weight: 500;
}

/* Cartões de métrica */
div[data-testid="metric-container"] {
    background: var(--secondary-background-color);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    transition: transform 0.2s;
}
div[data-testid="metric-container"]:hover {
    transform: translateY(-2px);
}
div[data-testid="metric-container"] label { color: #a0aec0 !important; font-size: 0.85rem; font-weight: 500; }
div[data-testid="metric-container"] [data-testid="metric-value"] {
    color: #00a651 !important;
    font-size: 1.8rem;
    font-weight: 700;
}

/* Cabeçalho hero */
.hero-header {
    background: linear-gradient(135deg, #006b33 0%, #00a651 50%, #00d669 100%);
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 24px;
    border: 1px solid #005a2b;
    box-shadow: 0 8px 24px rgba(0, 166, 81, 0.2);
}
.hero-header h1 { color: white; font-size: 2.4rem; font-weight: 800; margin: 0; letter-spacing: -1px; }
.hero-header p  { color: rgba(255,255,255,0.9); margin: 8px 0 0; font-size: 1.1rem; font-weight: 400; }

/* Badges de posição */
.badge-gol  { background:#1a4a33; color:#39d353; border: 1px solid #2ea043; }
.badge-lat  { background:#1a3a6b; color:#79c0ff; border: 1px solid #1f6feb; }
.badge-zag  { background:#3d4a1a; color:#a2e885; border: 1px solid #85e89d; }
.badge-mei  { background:#4a2e1a; color:#ffa657; border: 1px solid #d18c47; }
.badge-ata  { background:#5a1a1a; color:#ff7b72; border: 1px solid #da3633; }
.badge-tec  { background:#3a1a5a; color:#d2a8ff; border: 1px solid #8957e5; }
.pos-badge  {
    display:inline-block; padding:4px 12px;
    border-radius:20px; font-size:0.75rem; font-weight:700;
    letter-spacing:0.05em; text-transform: uppercase;
}

/* Tabela do time final */
.team-card {
    background: var(--secondary-background-color);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 20px;
    margin: 8px 0;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}
.team-card:hover { border-color: #00a651; transition: border-color .2s; }

/* Barra de progresso customizada */
.progress-bar-bg {
    background: rgba(255,255,255,0.05); border-radius: 8px; height: 8px; width: 100%;
}
.progress-bar-fill {
    background: linear-gradient(90deg, #006b33, #00a651);
    border-radius: 8px; height: 8px;
}

/* Alerts */
.alert-success { background:#1a3a2a; border:1px solid #39d353; border-radius:8px; padding:12px 16px; color:#aff5b4; }
.alert-warning { background:#3a2e1a; border:1px solid #ffa657; border-radius:8px; padding:12px 16px; color:#ffd8b0; }
.alert-error   { background:#3a1a1a; border:1px solid #ff7b72; border-radius:8px; padding:12px 16px; color:#ffc1bc; }
.alert-info    { background:#1a2a3a; border:1px solid #58a6ff; border-radius:8px; padding:12px 16px; color:#cae8ff; }

/* Separador */
hr.custom { border: none; border-top: 1px solid rgba(255,255,255,0.1); margin: 32px 0; }

/* Cards de instrução (Horizontais) */
.instruction-card {
    background: var(--secondary-background-color);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 24px;
    height: 100%;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
}
.instruction-icon {
    font-size: 2.5rem;
    margin-bottom: 16px;
    background: rgba(0, 166, 81, 0.1);
    width: 64px;
    height: 64px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 50%;
    color: #00a651;
}
.instruction-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 8px;
}
.instruction-text {
    font-size: 0.9rem;
    color: #a0aec0;
    line-height: 1.5;
}

/* Timeline Pipeline */
.timeline-container {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-top: 20px;
    position: relative;
    padding-bottom: 20px;
}
.timeline-container::before {
    content: '';
    position: absolute;
    top: 24px;
    left: 40px;
    right: 40px;
    height: 2px;
    background: rgba(255,255,255,0.1);
    z-index: 0;
}
.timeline-step {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    position: relative;
    z-index: 1;
    padding: 0 10px;
}
.timeline-icon {
    width: 48px;
    height: 48px;
    background: #1a1d27;
    border: 2px solid #00a651;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
    margin-bottom: 12px;
    box-shadow: 0 0 0 4px #0f1117; /* Fake gap with background color */
}
.timeline-title {
    font-weight: 700;
    font-size: 0.9rem;
    color: #ffffff;
    margin-bottom: 4px;
}
.timeline-desc {
    font-size: 0.75rem;
    color: #8b949e;
    line-height: 1.4;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
POSICAO_NOME  = {1: 'GOL', 2: 'LAT', 3: 'ZAG', 4: 'MEI', 5: 'ATA', 6: 'TEC'}
POSICAO_BADGE = {1:'badge-gol',2:'badge-lat',3:'badge-zag',4:'badge-mei',5:'badge-ata',6:'badge-tec'}

def badge_pos(pos_id: int) -> str:
    nome  = POSICAO_NOME.get(pos_id, '???')
    cls   = POSICAO_BADGE.get(pos_id, '')
    return f'<span class="pos-badge {cls}">{nome}</span>'

def fmt_preco(v): return f"C$ {v:.1f}"

def progress_html(pct):
    pct = min(100, max(0, pct))
    color = "#39d353" if pct >= 90 else "#ffa657" if pct >= 70 else "#58a6ff"
    return f"""
    <div class="progress-bar-bg">
        <div class="progress-bar-fill" style="width:{pct:.0f}%; background: {color};"></div>
    </div>
    <small style="color:#8b949e">{pct:.1f}%</small>"""

@st.cache_data(ttl=120, show_spinner=False)
def carregar_config():
    import yaml
    with open(ROOT_DIR / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_db_path():
    cfg = carregar_config()
    return ROOT_DIR / cfg.get("database", {}).get("path", "data/cartola.db")

@st.cache_data(ttl=300, show_spinner=False)
def info_banco():
    db = get_db_path()
    if not db.exists():
        return {}
    conn = sqlite3.connect(db)
    try:
        info = {}
        info['n_atletas']  = pd.read_sql("SELECT COUNT(*) as n FROM atletas", conn).iloc[0,0]
        info['n_pontuacoes'] = pd.read_sql("SELECT COUNT(*) as n FROM pontuacoes", conn).iloc[0,0]
        info['rodadas']    = pd.read_sql("SELECT DISTINCT rodada FROM pontuacoes ORDER BY rodada DESC LIMIT 1", conn)
        info['ultima_rodada'] = int(info['rodadas'].iloc[0,0]) if len(info['rodadas'])>0 else None
        conn.close()
        return info
    except:
        conn.close()
        return {}

# ─────────────────────────────────────────────────────────────
# SIDEBAR — CONFIGURAÇÕES
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path d="M12 2C6.48 2 2 6.48 2 12C2 17.52 6.48 22 12 22C17.52 22 22 17.52 22 12C22 6.48 17.52 2 12 2ZM12 4C16.41 4 20 7.59 20 12C20 13.56 19.55 15.01 18.79 16.27L15.27 12.75C15.73 11.96 16 11.02 16 10C16 7.79 14.21 6 12 6C9.79 6 8 7.79 8 10C8 11.02 8.27 11.96 8.73 12.75L5.21 16.27C4.45 15.01 4 13.56 4 12C4 7.59 7.59 4 12 4ZM12 8C13.1 8 14 8.9 14 10C14 11.1 13.1 12 12 12C10.9 12 10 11.1 10 10C10 8.9 10.9 8 12 8ZM6.64 17.7C7.65 18.84 9.06 19.58 10.63 19.89L11.75 16.71C11.16 16.5 10.64 16.14 10.25 15.68L6.64 17.7ZM17.36 17.7L13.75 15.68C13.36 16.14 12.84 16.5 12.25 16.71L13.37 19.89C14.94 19.58 16.35 18.84 17.36 17.7Z"/>
        </svg>
        <h2>CARTOLA FC</h2>
        <p>Optimizer AI</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Gerais", expanded=True):
        patrimonio = st.number_input(
            "Patrimônio (C$)", min_value=50.0, max_value=200.0, value=100.0, step=5.0
        )
        formacao = st.selectbox(
            "Formação Tática",
            ['4-3-3', '4-4-2', '3-5-2', '3-4-3', '4-5-1', '5-3-2', '5-4-1'],
            index=0
        )
        model_type = st.selectbox(
            "Modelo ML Predição",
            ['rf - Random Forest', 'gb - Gradient Boosting'],
            index=0
        )
        model_key = model_type.split(' ')[0]
        selected_strategy = st.selectbox(
            "Motor de Otimização",
            ['mega', 'genetic', 'ensemble'],
            index=1,
            help="Define o algoritmo que vai montar o time ideal."
        )

    with st.expander("Parâmetros Genéticos", expanded=False):
        cfg = carregar_config()
        opt_cfg = cfg.get('optimizer', {})

        population_size = st.slider("Tamanho da População",  100, 500, opt_cfg.get('population_size', 250), 50, help="Quantos times avaliar por geração. Valores altos encontram times melhores mas são mais lentos.")
        generations     = st.slider("Gerações",    50, 300, opt_cfg.get('generations', 150), 25, help="Qtd de iterações do algoritmo. Mais gerações = melhor otimização.")
        mutation_rate   = st.slider("Taxa de Mutação", 0.05, 0.40, opt_cfg.get('mutation_rate', 0.20), 0.05, help="Probabilidade de um jogador do time ser trocado por outro aleatório.")
        max_mesmo_clube = st.slider("Máx. Jogadores do mesmo Clube", 1, 5, opt_cfg.get('max_mesmo_clube', 2), 1, help="Evita times inteiros de uma só equipe, mitigando risco de revés.")

    with st.expander("Status da API / Banco", expanded=False):
        info = info_banco()
        if info:
            st.markdown(f"""
            <div style="background:var(--secondary-background-color); border-radius:8px; padding:12px; border:1px solid rgba(255,255,255,0.08);">
                <div style="color:#a0aec0; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.5px;">Atletas na base</div>
                <div style="color:#00a651; font-weight:800; font-size:1.4rem;">{info.get('n_atletas', 0):,}</div>
                <div style="color:#a0aec0; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.5px; margin-top:12px;">Última rodada ref.</div>
                <div style="color:#ffffff; font-weight:800; font-size:1.4rem;">#{info.get('ultima_rodada', '—')}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert-warning">⚠️ Banco de dados não encontrado.</div>', unsafe_allow_html=True)
            
    st.markdown("<br>", unsafe_allow_html=True)
    realizar_analise_sidebar = st.button("▶ Executar Análise", type="primary", use_container_width=True)

# ─────────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <h1>Cartola FC · Optimizer AI</h1>
    <p>Otimize sua escalação usando Machine Learning, Pesos Táticos e Algoritmos de Elite.</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# ABAS PRINCIPAIS
# ─────────────────────────────────────────────────────────────
tab_home, tab_analise, tab_time, tab_historico = st.tabs([
    "🏠 Início",
    "📊 Análise da Rodada",
    "🏆 Time Otimizado",
    "📈 Histórico"
])

# ══════════════════════════════════════════════════════════════
# ABA INÍCIO
# ══════════════════════════════════════════════════════════════
with tab_home:
    st.markdown("### Visão Geral do Sistema")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Gerações", generations, delta="Ótimo" if generations >= 150 else "Baixo")
    with col2:
        st.metric("População", population_size, delta="Preciso" if population_size >= 250 else "Rápido")
    with col3:
        st.metric("Máx. Jogadores/Clube", max_mesmo_clube, delta="-Risco" if max_mesmo_clube <= 3 else "+Risco", delta_color="inverse")
    with col4:
        val = info.get('ultima_rodada', '—')
        st.metric("Última Rodada no BD", f"#{val}" if val != '—' else '—', delta="Atualizado")

    st.markdown("<hr class='custom'>", unsafe_allow_html=True)

    st.markdown("### Onde a Magia Acontece 🚀")
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("""
        <div class="instruction-card">
            <div class="instruction-icon">📡</div>
            <div class="instruction-title">Coleta em Tempo Real</div>
            <div class="instruction-text">Buscamos dados de mercado, atletas prováveis e partidas da rodada diretamente da API oficial da Globo.</div>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class="instruction-card">
            <div class="instruction-icon">🧠</div>
            <div class="instruction-title">Estatística + ML</div>
            <div class="instruction-text">Aplicamos modelos (RF/GBM) no histórico buscando padrões e geramos 18+ features (mando, fase, probabilidade de SG).</div>
        </div>
        """, unsafe_allow_html=True)
    with col_c:
        st.markdown("""
        <div class="instruction-card">
            <div class="instruction-icon">🧬</div>
            <div class="instruction-title">Genética & Tática</div>
            <div class="instruction-text">Usamos inteligência artificial evolutiva para combinar milhões de escalações até encontrar a matemática perfeita pro seu orçamento.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr class='custom'>", unsafe_allow_html=True)
    
    st.markdown("### Pipeline de Execução Passo a Passo")
    st.markdown("""
    <div class="timeline-container">
        <div class="timeline-step">
            <div class="timeline-icon">1</div>
            <div class="timeline-title">Download API</div>
            <div class="timeline-desc">Mercado e Status</div>
        </div>
        <div class="timeline-step">
            <div class="timeline-icon">2</div>
            <div class="timeline-title">Feature Engineering</div>
            <div class="timeline-desc">Transformação de Dados</div>
        </div>
        <div class="timeline-step">
            <div class="timeline-icon">3</div>
            <div class="timeline-title">Treinamento ML</div>
            <div class="timeline-desc">Previsão de Pontos</div>
        </div>
        <div class="timeline-step">
            <div class="timeline-icon">4</div>
            <div class="timeline-title">Estratégia Tática</div>
            <div class="timeline-desc">Multiplicadores do Especialista</div>
        </div>
        <div class="timeline-step">
            <div class="timeline-icon">5</div>
            <div class="timeline-title">AG Otimizador</div>
            <div class="timeline-desc">Evolução do Time</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<hr class='custom'>", unsafe_allow_html=True)
    
    st.markdown("### Características Táticas (Pesos)")
    tactics_data = [
        {"Condição": "Atacante x Defesa Fraca", "Impacto": "+30% Pontos"},
        {"Condição": "Meia x Defesa Fraca", "Impacto": "+25% Pontos"},
        {"Condição": "Qualquer Posição em Casa", "Impacto": "+10% Pontos"},
        {"Condição": "Zagueiro com alta prob. de SG", "Impacto": "+15% Pontos"},
        {"Condição": "Mais de 3 jogadores do mesmo clube", "Impacto": "Penalidade Severa"},
        {"Condição": "Jogadores de times adversários (mesmo jogo)", "Impacto": "Penalidade Severa (Anti-Confronto)"},
    ]
    tactics_df = pd.DataFrame(tactics_data)
    
    def highlight_impact(val: str) -> str:
        if "+" in str(val):
            return 'color: #00d669; font-weight: bold'
        elif "Penalidade" in str(val):
            return 'color: #ff7b72; font-weight: bold'
        return ''
        
    st.dataframe(
        tactics_df.style.applymap(highlight_impact, subset=['Impacto']),
        use_container_width=True,
        hide_index=True
    )

# ══════════════════════════════════════════════════════════════
# ABA ANÁLISE DA RODADA
# ══════════════════════════════════════════════════════════════
with tab_analise:

    st.markdown("ℹ️ *A análise central agora é iniciada pelo botão 'Executar Análise' na barra lateral.*")

    if realizar_analise_sidebar:
        with st.spinner("Inicializando pipeline..."):

            progress = st.progress(0, text="🔄 Preparando...")
            log_area = st.empty()
            logs = []

            def log(msg: str, tipo: str = "info") -> None:
                emoji = {"info":"ℹ️","ok":"✅","warn":"⚠️","err":"❌"}.get(tipo,"ℹ️")
                ts = datetime.now().strftime("%H:%M:%S")
                logs.append(f"`{ts}` {emoji} {msg}")
                logging.info(f"[{tipo.upper()}] {msg}")
                log_area.markdown("\n\n".join(logs[-12:]))

            try:
                from dotenv import load_dotenv
                load_dotenv(ROOT_DIR / ".env")

                EMAIL     = os.getenv('CARTOLA_EMAIL')
                PASSWORD  = os.getenv('CARTOLA_PASSWORD')
                PATRIMONIO = patrimonio
                FORMACAO   = formacao

                # ── 1. API Client ──────────────────────────────
                log("Inicializando cliente API...")
                progress.progress(5, "📡 Conectando à API...")

                from src.api.client import CartolaAPIClient
                api_client = CartolaAPIClient(email=EMAIL, password=PASSWORD)
                log("Cliente API pronto", "ok")

                # ── 2. Coletar dados ───────────────────────────
                from src.data.collector import CartolaDataCollector
                collector = CartolaDataCollector(api_client)
                progress.progress(15, "📥 Coletando mercado...")

                mercado = collector.collect_mercado_status()
                from src.utils.validators import validar_mercado, validar_formacao, filtrar_atletas_com_jogo
                mercado_info = validar_mercado(mercado)
                if not mercado_info['valido']:
                    st.error(f"❌ {mercado_info['mensagem']}")
                    st.stop()

                rodada_atual = mercado_info['rodada_atual']
                log(f"Rodada {rodada_atual} — mercado {'ABERTO' if mercado_info.get('mercado_aberto') else 'FECHADO'}", "ok")
                progress.progress(25, "👥 Coletando atletas...")

                atletas_df = collector.collect_atletas_mercado(rodada_atual)
                log(f"{len(atletas_df)} atletas coletados", "ok")

                # Salvar mapa de clubes e partidas no estado da sessão
                _raw_mercado = api_client.get_atletas_mercado()
                _clubes_raw  = _raw_mercado.get('clubes', {})
                clubes_map_full = {
                    int(k): v.get('nome', v.get('abreviacao', str(k)))
                    for k, v in _clubes_raw.items()
                } if isinstance(_clubes_raw, dict) else {}
                abrev_map = {
                    int(k): v.get('abreviacao', v.get('nome', str(k))[:3].upper())
                    for k, v in _clubes_raw.items()
                } if isinstance(_clubes_raw, dict) else {}
                st.session_state['clubes_map']  = clubes_map_full
                st.session_state['abrev_map']   = abrev_map

                collector.collect_partidas(rodada_atual)
                progress.progress(35, "📚 Carregando histórico...")

                # ── 3. Histórico ───────────────────────────────
                conn = sqlite3.connect(str(collector.db_path))
                ano_atual = mercado.get('temporada', 2026)
                
                historico_df = pd.read_sql_query("""
                    SELECT p.*, a.clube_id FROM pontuacoes p
                    JOIN atletas a ON p.atleta_id = a.atleta_id
                    WHERE p.ano = ? AND p.rodada >= ? AND p.rodada <= ?
                    ORDER BY p.atleta_id, p.rodada
                """, conn, params=(ano_atual, max(1, rodada_atual - 15), rodada_atual))

                partidas_df = pd.read_sql_query("""
                    SELECT * FROM partidas WHERE ano = ? AND rodada >= ? AND rodada <= ?
                """, conn, params=(ano_atual, max(1, rodada_atual - 15), rodada_atual))
                conn.close()

                # Mapa clube_id → partida da rodada (mandante vs visitante)
                _pm = {}
                for _, _pr in partidas_df[partidas_df['rodada'] == rodada_atual].iterrows():
                    _cid_a = int(_pr['clube_casa_id']) if _pr['clube_casa_id'] else 0
                    _cid_b = int(_pr['clube_visitante_id']) if _pr['clube_visitante_id'] else 0
                    _ab    = abrev_map
                    _jogo  = f"{_ab.get(_cid_a, str(_cid_a))} 🏟️ vs {_ab.get(_cid_b, str(_cid_b))}"
                    _pm[_cid_a] = _jogo
                    _pm[_cid_b] = _jogo
                st.session_state['partidas_map'] = _pm

                log(f"{len(historico_df)} registros de pontuação | {len(partidas_df)} partidas", "ok")

                # Filtrar sem jogo
                atletas_df = filtrar_atletas_com_jogo(atletas_df, rodada_atual, partidas_df)
                log(f"{len(atletas_df)} atletas com jogo confirmado na rodada {rodada_atual}", "ok")

                progress.progress(45, "🔧 Feature engineering...")

                # ── 4. Features ────────────────────────────────
                from src.ml.features import FeatureEngineer
                if len(historico_df) > 0:
                    historico_df = FeatureEngineer.engineer_all_features(historico_df, partidas_df)
                    n_features = historico_df.shape[1]
                    log(f"Feature engineering concluído: {n_features} colunas", "ok")
                else:
                    log("Sem histórico para feature engineering", "warn")

                progress.progress(55, "🧠 Treinando modelo ML...")

                # ── 5. Treinar ─────────────────────────────────
                from src.ml.predictor import CartolaPredictor, ValorizacaoPredictor
                predictor = CartolaPredictor(model_type=model_key)
                val_predictor = ValorizacaoPredictor(model_type='gb')

                dados_treino = historico_df[historico_df['rodada'] < rodada_atual] if len(historico_df) > 0 else pd.DataFrame()
                can_train    = len(dados_treino) >= CartolaPredictor.HISTORICO_MINIMO

                if can_train:
                    metrics = predictor.train(dados_treino, validate=True)
                    log(f"Modelo treinado — MAE: {metrics['mae']:.2f} | R²: {metrics['r2']:.3f}", "ok")

                    # Treinar Valorização
                    val_samples = FeatureEngineer.create_valorizacao_samples(historico_df)
                    if len(val_samples) >= ValorizacaoPredictor.HISTORICO_MINIMO:
                        val_metrics = val_predictor.train(val_samples, validate=False)
                        log(f"Modelo Valorização — MAE: {val_metrics['mae']:.2f} | R²: {val_metrics['r2']:.3f}", "ok")
                    else:
                        log("Histórico insuficiente para Treinar Valorização.", "warn")

                else:
                    log(f"Histórico insuficiente ({len(dados_treino)} registros). Usando heurística.", "warn")

                progress.progress(68, "🎯 Predizendo próxima rodada...")

                # ── 6. Predição ────────────────────────────────
                if can_train and predictor.is_trained:
                    ultimos = historico_df.sort_values('rodada').groupby('atleta_id').tail(5)
                    ultimos = ultimos.copy()
                    ultimos['pontos_ewm_local'] = (
                        ultimos.groupby('atleta_id')['pontos']
                        .transform(lambda x: x.ewm(span=3, min_periods=1).mean())
                    )
                    ultimos_agg = ultimos.sort_values('rodada').groupby('atleta_id').last().reset_index()

                    dados_pred = atletas_df.merge(
                        ultimos_agg, on='atleta_id', how='left', suffixes=('', '_hist')
                    ).fillna(0)

                    tem_taticas = ('bonus_oponente_fraco' in dados_pred.columns and
                                   'mando_casa' in dados_pred.columns)

                    if tem_taticas:
                        predicoes_df = predictor.predict_with_tactical_weights(dados_pred)
                        log("Predições com pesos táticos aplicados", "ok")
                    else:
                        predicoes_df = predictor.predict_with_confidence(dados_pred)
                        predicoes_df['predicao_ajustada'] = predicoes_df['predicao']
                        log("Predição base (features táticas indisponíveis)", "warn")

                    # Prever valorização
                    if val_predictor.is_trained:
                        val_pred_df = val_predictor.predict(dados_pred)
                        predicoes_df['esperanca_valorizacao'] = val_pred_df['esperanca_valorizacao']
                        log("Predição de Valorização (Cartoletas) gerada", "ok")
                    else:
                        predicoes_df['esperanca_valorizacao'] = 0.0

                else:
                    predicoes_df = CartolaPredictor.fallback_heuristica(atletas_df)
                    predicoes_df['predicao_ajustada'] = predicoes_df['predicao']
                    predicoes_df['esperanca_valorizacao'] = 0.0

                log(f"Predições geradas para {len(predicoes_df)} atletas", "ok")
                progress.progress(82, "🧬 Otimizando time genético...")

                # ── 7. Otimizador Factory ──────────────────────────────
                from src.optimizer.factory import CartolaOptimizer

                predicoes_opt = predicoes_df.copy()
                if 'predicao_std' not in predicoes_opt.columns:
                    predicoes_opt['predicao_std'] = 0.0
                if 'esperanca_valorizacao' not in predicoes_opt.columns:
                    predicoes_opt['esperanca_valorizacao'] = 0.0
                score_col = 'predicao_ajustada' if 'predicao_ajustada' in predicoes_opt.columns else 'predicao'
                predicoes_opt['predicao'] = predicoes_opt[score_col]

                # Juntar para o formato que a factory e outras strategies esperam
                df_for_opt = atletas_df.merge(predicoes_opt[['atleta_id', 'predicao', 'predicao_std', 'esperanca_valorizacao']], on='atleta_id', how='left')
                df_for_opt['mega_score'] = df_for_opt['predicao']

                opt_config = {
                    'population_size': population_size,
                    'generations': generations,
                    'mutation_rate': mutation_rate,
                    'max_mesmo_clube': max_mesmo_clube,
                    'test_all_formations': False,
                    'strategies': ['mega', 'genetic']
                }

                optimizer = CartolaOptimizer(strategy=selected_strategy, config=opt_config)
                # Passar partidas da rodada para ativar regra anti-confronto
                _partidas_rodada_df = partidas_df[partidas_df['rodada'] == rodada_atual] if len(partidas_df) > 0 else pd.DataFrame()
                lineup = optimizer.optimize(df_for_opt, PATRIMONIO, FORMACAO, partidas_df=_partidas_rodada_df)

                if lineup is None or len(lineup) < 12:
                     raise Exception(f"Não foi possível gerar time viável com a estratégia '{selected_strategy}'. Tente relaxar os filtros.")

                best_team = lineup.to_dict('records')
                
                val_score = float(lineup['predicao'].sum()) if 'predicao' in lineup.columns else float(lineup['mega_score'].sum())
                val_preco = float(lineup['preco'].sum())
                
                stats = {
                    'total_pontos_preditos': val_score,
                    'total_preco': val_preco,
                    'patrimonio_usado': (val_preco / PATRIMONIO) * 100 if PATRIMONIO > 0 else 0,
                }

                # Formatar
                from src.ml.optimizer import POSICOES
                display_df = lineup.copy()
                if 'posicao_nome' not in display_df.columns:
                    display_df['posicao_nome'] = display_df['posicao_id'].map(POSICOES)
                
                cols_to_show = []
                for c in ['apelido', 'posicao_nome', 'clube_id', 'preco', 'predicao', 'media']:
                    if c in display_df.columns:
                        cols_to_show.append(c)
                team_df = display_df[cols_to_show]

                log(f"Time otimizado ({selected_strategy}): {stats['total_pontos_preditos']:.1f} pts | C$ {stats['total_preco']:.1f}", "ok")

                # Salvar CSV
                out = ROOT_DIR / "data" / "processed"
                out.mkdir(parents=True, exist_ok=True)
                team_df.to_csv(out / "time_sugerido.csv", index=False)
                predicoes_df.to_csv(out / "predicoes_rodada.csv", index=False)

                progress.progress(100, "✅ Concluído!")

                # ── Persistir no session_state ─────────────────
                st.session_state['predicoes_df'] = predicoes_df
                st.session_state['team_df']      = team_df
                st.session_state['best_team']    = best_team
                st.session_state['stats']        = stats
                st.session_state['rodada']       = rodada_atual
                st.session_state['atletas_df']   = atletas_df
                st.session_state['executado']    = True
                st.session_state['historico_df'] = historico_df

                st.markdown('<div class="alert-success">✅ Pipeline concluído com sucesso! Veja as abas <b>Análise</b> e <b>Time Otimizado</b>.</div>', unsafe_allow_html=True)

            except sqlite3.Error as db_e:
                logging.error(f"Erro de Banco de Dados: {db_e}", exc_info=True)
                st.markdown(f'<div class="alert-error">🗄️ Erro ao acessar a Base de Dados do Cartola: {db_e}</div>', unsafe_allow_html=True)
            except ValueError as ve:
                logging.error(f"Erro de Validação de Dados: {ve}", exc_info=True)
                st.markdown(f'<div class="alert-error">⚠️ Erro nos dados processados: {ve}</div>', unsafe_allow_html=True)
            except Exception as e:
                logging.error(f"Falha inesperada no pipeline: {e}", exc_info=True)
                import traceback
                st.markdown(f'<div class="alert-error">❌ Erro Inesperado no Pipeline: {e}</div>', unsafe_allow_html=True)
                with st.expander("🔍 Detalhes do erro (Modo Desenvolvedor)"):
                    st.code(traceback.format_exc())

    # ── Resultado da análise (persiste por sessão) ─────────
    if st.session_state.get('executado'):
        predicoes_df = st.session_state['predicoes_df']
        rodada_atual  = st.session_state['rodada']

        st.markdown("<hr class='custom'>", unsafe_allow_html=True)
        st.success("✅ Análise e otimização concluídas com sucesso! Clique na aba **🏆 Time Otimizado** acima para ver os 12 jogadores escalados.")

        score_col = 'predicao_ajustada' if 'predicao_ajustada' in predicoes_df.columns else 'predicao'
        pred_filtrado = predicoes_df.sort_values(score_col, ascending=False).head(50)

        # Download
        csv_bytes = pred_filtrado.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️  Baixar predições (.csv)", csv_bytes,
            file_name=f"predicoes_rodada_{rodada_atual}.csv",
            mime="text/csv", use_container_width=True
        )

# ══════════════════════════════════════════════════════════════
# ABA TIME OTIMIZADO
# ══════════════════════════════════════════════════════════════
with tab_time:
    if not st.session_state.get('executado'):
        st.markdown('<div class="alert-info">ℹ️ Execute a análise na aba <b>Análise da Rodada</b> para ver o time otimizado.</div>', unsafe_allow_html=True)
    else:
        team_df   = st.session_state['team_df']
        best_team = st.session_state['best_team']
        stats     = st.session_state['stats']
        rodada    = st.session_state['rodada']

        # ── KPIs ───────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("⚽ Pts Preditos", f"{stats['total_pontos_preditos']:.1f}")
        c2.metric("💰 Custo Total", fmt_preco(stats['total_preco']))
        c3.metric("📊 Patrimônio Usado", f"{stats['patrimonio_usado']:.1f}%")
        c4.metric("📅 Rodada", f"#{rodada}")

        # Barra de uso do patrimônio
        st.markdown(progress_html(stats['patrimonio_usado']), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # ── Time por posição ───────────────────────────────
        st.markdown(f"### 🏆 Escalação Sugerida — {formacao}")

        # Recuperar mapas de clube e partida
        _abrev_map    = st.session_state.get('abrev_map', {})
        _clubes_full  = st.session_state.get('clubes_map', {})
        _partidas_map = st.session_state.get('partidas_map', {})

        POS_ORDER = {1:1, 2:3, 3:2, 4:4, 5:5, 6:6}  # GOL, ZAG, LAT, MEI, ATA, TEC
        pos_order_rev = {1:'⛔ Goleiro', 2:'🛡️ Laterais', 3:'🛡️ Zagueiros',
                         4:'⚙️ Meias', 5:'🔥 Atacantes', 6:'📋 Técnico'}

        atletas_sorted = sorted(best_team, key=lambda a: POS_ORDER.get(a.get('posicao_id', 9), 9))

        # Cabeçalho da tabela
        hdr = st.columns([2.8, 1.4, 1.2, 2.2, 2.2, 1.8])
        for _col, _lbl in zip(hdr, ['JOGADOR', 'CLUBE', 'PREÇO', 'PARTIDA', 'PREDIÇÃO', 'TAGS']):
            _col.markdown(f"<span style='color:#8b949e; font-size:0.75rem; font-weight:600; letter-spacing:.05em'>{_lbl}</span>",
                          unsafe_allow_html=True)
        st.markdown("<hr style='border-color:#21262d; margin:4px 0 8px'>", unsafe_allow_html=True)

        current_pos = None
        capitao_dict = max(atletas_sorted, key=lambda x: x.get('predicao_ajustada', x.get('predicao', 0))) if atletas_sorted else {}

        for atleta in atletas_sorted:
            pos_id   = atleta.get('posicao_id', 0)
            clube_id = int(atleta.get('clube_id', 0))
            is_capitao = (atleta == capitao_dict)

            if pos_id != current_pos:
                current_pos = pos_id
                st.markdown(
                    f"<div style='margin:12px 0 4px; color:#58a6ff; font-weight:700; font-size:0.82rem; letter-spacing:.06em;'>{pos_order_rev.get(pos_id, '?')}</div>",
                    unsafe_allow_html=True
                )

            nome    = atleta.get('apelido', atleta.get('nome', '—'))
            preco   = atleta.get('preco', 0)
            pred    = atleta.get('predicao', 0)
            media   = atleta.get('media', 0)
            boost   = atleta.get('bonus_oponente_fraco', 0)
            em_casa = atleta.get('mando_casa', 0)
            val     = atleta.get('esperanca_valorizacao', 0)

            # Clube
            clube_abrev = _abrev_map.get(clube_id, str(clube_id))
            clube_full  = _clubes_full.get(clube_id, clube_abrev)

            # Partida
            jogo_str = _partidas_map.get(clube_id, '—')

            # Tags
            boost_tags = ""
            if is_capitao:
                boost_tags += '<span class="boost-tag" style="color:#ffd700; border-color:#ffd700">👑Capitão</span> '
            if val > 0.5:
                boost_tags += f'<span class="boost-tag" style="color:#39d353; border-color:#39d353">🤑+C${val:.1f}</span> '
            elif val < -0.5:
                boost_tags += f'<span class="boost-tag" style="color:#ff7b72; border-color:#ff7b72">📉C${val:.1f}</span> '
            if boost > 0.7 and pos_id in [4, 5]:
                boost_tags += '<span class="boost-tag">🎯Def.Fraca</span> '
            if em_casa:
                boost_tags += '<span class="boost-tag">🏠Casa</span> '
            if pred > 8:
                boost_tags += '<span class="boost-tag">⭐Elite</span> '

            pct_bar = min(100, (pred / 12) * 100)
            col_nome, col_clube, col_preco, col_jogo, col_pred, col_extra = st.columns([2.8, 1.4, 1.2, 2.2, 2.2, 1.8])

            with col_nome:
                st.markdown(
                    f"{badge_pos(pos_id)}&nbsp;<span style='font-weight:600'>{nome}</span>"
                    f"<br><span style='color:#8b949e; font-size:0.72rem;'>Média: {media:.1f}</span>",
                    unsafe_allow_html=True
                )
            with col_clube:
                st.markdown(
                    f"<span style='background:#21262d; border:1px solid #30363d; border-radius:6px;"
                    f" padding:3px 8px; font-size:0.8rem; font-weight:700; color:#e6edf3;'"
                    f" title='{clube_full}'>{clube_abrev}</span>",
                    unsafe_allow_html=True
                )
            with col_preco:
                st.markdown(
                    f"<span style='color:#ffa657; font-weight:700;'>C$ {preco:.1f}</span>",
                    unsafe_allow_html=True
                )
            with col_jogo:
                st.markdown(
                    f"<span style='color:#8b949e; font-size:0.8rem;'>{jogo_str}</span>",
                    unsafe_allow_html=True
                )
            with col_pred:
                st.markdown(f"""
                <div style="display:flex; align-items:center; gap:8px;">
                    <div class="progress-bar-bg" style="flex:1">
                        <div class="progress-bar-fill" style="width:{pct_bar:.0f}%"></div>
                    </div>
                    <span style="color:#e6edf3; font-weight:700; min-width:46px;">{pred:.1f}pts</span>
                </div>""", unsafe_allow_html=True)
            with col_extra:
                st.markdown(boost_tags or "<span style='color:#444'>—</span>", unsafe_allow_html=True)

        # ── Comparativo com médias ─────────────────────────
        st.markdown("<hr class='custom'>", unsafe_allow_html=True)
        st.markdown("#### 📊 Distribuição por Posição")

        _abrev_map2   = st.session_state.get('abrev_map', {})
        _partidas_map2 = st.session_state.get('partidas_map', {})
        rows_pos = []
        pos_groups = {}
        for a in best_team:
            pid = a.get('posicao_id', 0)
            pos_groups.setdefault(pid, []).append(a)

        for pid, atletas in sorted(pos_groups.items()):
            nomes  = ", ".join(a.get('apelido', '?') for a in atletas)
            clubes = ", ".join(_abrev_map2.get(int(a.get('clube_id', 0)), '?') for a in atletas)
            jogos  = ", ".join(_partidas_map2.get(int(a.get('clube_id', 0)), '—') for a in atletas)
            total  = sum(a.get('predicao', 0) for a in atletas)
            custo  = sum(a.get('preco', 0) for a in atletas)
            rows_pos.append({
                'Posição': POSICAO_NOME.get(pid, '?'),
                'Atletas': nomes,
                'Clube(s)': clubes,
                'Partida(s)': jogos,
                'Pts Preditos': f"{total:.1f}",
                'Custo C$': f"{custo:.1f}",
            })

        st.dataframe(pd.DataFrame(rows_pos), use_container_width=True, hide_index=True)

        # ── Download ───────────────────────────────────────
        st.markdown("<hr class='custom'>", unsafe_allow_html=True)
        csv_time = team_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️ Baixar time (.csv)", csv_time,
            file_name=f"time_rodada_{rodada}.csv",
            mime="text/csv", use_container_width=True
        )

# ══════════════════════════════════════════════════════════════
# ABA HISTÓRICO
# ══════════════════════════════════════════════════════════════
with tab_historico:
    db = get_db_path()
    if not db.exists():
        st.markdown('<div class="alert-warning">⚠️ Banco de dados não encontrado. Execute a análise primeiro.</div>', unsafe_allow_html=True)
    else:
        conn = sqlite3.connect(str(db))

        st.markdown("#### 📊 Pontuações por Rodada")
        rodadas_df = pd.read_sql("""
            SELECT rodada, COUNT(DISTINCT atleta_id) as n_atletas,
                   AVG(pontos) as media_pontos, MAX(pontos) as max_pontos
            FROM pontuacoes GROUP BY rodada ORDER BY rodada DESC
        """, conn)

        if len(rodadas_df) > 0:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Rodadas com dados**")
                st.dataframe(rodadas_df.rename(columns={
                    'rodada':'Rodada','n_atletas':'Atletas',
                    'media_pontos':'Média Pts','max_pontos':'Máx Pts'
                }), use_container_width=True, hide_index=True)
            with c2:
                if len(rodadas_df) > 1:
                    st.markdown("**Evolução da média de pontos**")
                    chart_df = rodadas_df.sort_values('rodada').set_index('rodada')['media_pontos']
                    st.line_chart(chart_df)
        else:
            st.markdown('<div class="alert-info">ℹ️ Nenhuma rodada encontrada no banco.</div>', unsafe_allow_html=True)

        st.markdown("<hr class='custom'>", unsafe_allow_html=True)
        st.markdown("#### 🔍 Buscar Atleta")

        atleta_nome = st.text_input("Nome ou apelido do atleta")
        if atleta_nome:
            resultados = pd.read_sql("""
                SELECT a.apelido, a.clube_id, p.rodada, p.pontos, p.preco, p.media,
                       p.G, p.A, p.SG, p.DS, p.FC, p.CA
                FROM pontuacoes p
                JOIN atletas a ON p.atleta_id = a.atleta_id
                WHERE a.apelido LIKE ? OR a.nome LIKE ?
                ORDER BY a.apelido, p.rodada DESC
                LIMIT 100
            """, conn, params=(f"%{atleta_nome}%", f"%{atleta_nome}%"))

            if len(resultados) > 0:
                st.success(f"✅ {len(resultados)} registros encontrados")
                st.dataframe(resultados, use_container_width=True, hide_index=True)
            else:
                st.warning("Nenhum atleta encontrado com esse nome.")

        conn.close()
