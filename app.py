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

/* Fundo escuro */
.stApp { background: #0d1117; color: #e6edf3; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #30363d;
}

/* Cartões de métrica */
div[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 16px 20px;
}
div[data-testid="metric-container"] label { color: #8b949e !important; font-size: 0.78rem; }
div[data-testid="metric-container"] [data-testid="metric-value"] {
    color: #58a6ff !important;
    font-size: 1.6rem;
    font-weight: 700;
}

/* Cabeçalho hero */
.hero-header {
    background: linear-gradient(135deg, #1a7fe8 0%, #0d47a1 50%, #1565c0 100%);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 20px;
    border: 1px solid #1a4db5;
}
.hero-header h1 { color: white; font-size: 2.2rem; font-weight: 700; margin: 0; }
.hero-header p  { color: rgba(255,255,255,0.8); margin: 4px 0 0; font-size: 1rem; }

/* Badges de posição */
.badge-gol  { background:#1a6b46; color:#39d353; }
.badge-lat  { background:#1a3a6b; color:#79c0ff; }
.badge-zag  { background:#2e3a1a; color:#85e89d; }
.badge-mei  { background:#4a2e1a; color:#ffa657; }
.badge-ata  { background:#5a1a1a; color:#ff7b72; }
.badge-tec  { background:#3a1a5a; color:#d2a8ff; }
.pos-badge  {
    display:inline-block; padding:2px 10px;
    border-radius:20px; font-size:0.75rem; font-weight:600;
    letter-spacing:0.05em;
}

/* Tabela do time final */
.team-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px;
    margin: 8px 0;
}
.team-card:hover { border-color: #58a6ff; transition: border-color .2s; }

/* Barra de progresso customizada */
.progress-bar-bg {
    background: #21262d; border-radius: 8px; height: 8px; width: 100%;
}
.progress-bar-fill {
    background: linear-gradient(90deg, #1a7fe8, #58a6ff);
    border-radius: 8px; height: 8px;
}

/* Alerts */
.alert-success { background:#1a3a2a; border:1px solid #39d353; border-radius:8px; padding:12px 16px; color:#aff5b4; }
.alert-warning { background:#3a2e1a; border:1px solid #ffa657; border-radius:8px; padding:12px 16px; color:#ffd8b0; }
.alert-error   { background:#3a1a1a; border:1px solid #ff7b72; border-radius:8px; padding:12px 16px; color:#ffc1bc; }
.alert-info    { background:#1a2a3a; border:1px solid #58a6ff; border-radius:8px; padding:12px 16px; color:#cae8ff; }

/* Separador */
hr.custom { border: none; border-top: 1px solid #30363d; margin: 24px 0; }

/* Boost tags */
.boost-tag {
    background: #1a3a2a; color: #39d353;
    border-radius: 6px; padding: 1px 8px;
    font-size: 0.72rem; font-weight: 600;
}
.penalidade-tag {
    background: #3a1a1a; color: #ff7b72;
    border-radius: 6px; padding: 1px 8px;
    font-size: 0.72rem; font-weight: 600;
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
    <div style="text-align:center; padding: 16px 0 8px;">
        <span style="font-size:2.5rem">⚽</span>
        <h2 style="color:#58a6ff; margin:4px 0 0; font-size:1.2rem; font-weight:700;">CARTOLA FC</h2>
        <p style="color:#8b949e; font-size:0.8rem; margin:0;">Optimizer AI</p>
    </div>
    <hr style="border-color:#30363d; margin:12px 0 20px;">
    """, unsafe_allow_html=True)

    st.markdown("##### ⚙️ Configurações")

    patrimonio = st.number_input(
        "💰 Patrimônio (C$)", min_value=50.0, max_value=200.0, value=100.0, step=5.0
    )

    formacao = st.selectbox(
        "⚽ Formação",
        ['4-3-3', '4-4-2', '3-5-2', '3-4-3', '4-5-1', '5-3-2', '5-4-1'],
        index=0
    )

    model_type = st.selectbox(
        "🧠 Modelo ML",
        ['rf - Random Forest', 'gb - Gradient Boosting'],
        index=0
    )
    model_key = model_type.split(' ')[0]

    st.markdown("<hr style='border-color:#30363d; margin:16px 0;'>", unsafe_allow_html=True)
    st.markdown("##### 🧬 Otimizador Genético")

    cfg = carregar_config()
    opt_cfg = cfg.get('optimizer', {})

    population_size = st.slider("População",  100, 500, opt_cfg.get('population_size', 250), 50)
    generations     = st.slider("Gerações",    50, 300, opt_cfg.get('generations', 150), 25)
    mutation_rate   = st.slider("Taxa Mutação", 0.05, 0.40, opt_cfg.get('mutation_rate', 0.20), 0.05)
    max_mesmo_clube = st.slider("Máx. por Clube", 1, 5, opt_cfg.get('max_mesmo_clube', 3), 1)

    st.markdown("<hr style='border-color:#30363d; margin:16px 0;'>", unsafe_allow_html=True)
    st.markdown("##### 🌐 API / Banco")

    info = info_banco()
    if info:
        st.markdown(f"""
        <div style="background:#161b22; border-radius:8px; padding:12px; border:1px solid #30363d;">
            <div style="color:#8b949e; font-size:0.78rem;">Atletas no banco</div>
            <div style="color:#58a6ff; font-weight:700; font-size:1.1rem;">{info.get('n_atletas', 0):,}</div>
            <div style="color:#8b949e; font-size:0.78rem; margin-top:8px;">Pontuações registradas</div>
            <div style="color:#58a6ff; font-weight:700; font-size:1.1rem;">{info.get('n_pontuacoes', 0):,}</div>
            <div style="color:#8b949e; font-size:0.78rem; margin-top:8px;">Última rodada</div>
            <div style="color:#39d353; font-weight:700; font-size:1.1rem;">Rodada {info.get('ultima_rodada', '—')}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert-warning">⚠️ Banco de dados vazio ou não encontrado.</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <h1>⚽ Cartola FC · Optimizer AI</h1>
    <p>Análise completa da rodada com Machine Learning + Algoritmo Genético + Pesos Táticos</p>
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
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🧬 Gerações", generations)
    with col2:
        st.metric("👥 População", population_size)
    with col3:
        st.metric("🏟️ Máx. Clube", max_mesmo_clube)
    with col4:
        val = info.get('ultima_rodada', '—')
        st.metric("📅 Última Rodada", f"#{val}" if val != '—' else '—')

    st.markdown("<hr class='custom'>", unsafe_allow_html=True)

    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.markdown("#### 🚀 Como usar")
        st.markdown("""
        1. **Configure** o patrimônio, formação e modelo na barra lateral
        2. Clique em **▶ Executar Análise** na aba *Análise da Rodada*
        3. O sistema irá:
           - 📡 Coletar dados da API do Cartola FC
           - 🔧 Criar **features táticas** (mando, força do adversário, momento)
           - 🧠 Treinar modelo ML com histórico das últimas rodadas
           - 🎯 Aplicar **pesos táticos** por posição e contexto
           - 🧬 Otimizar o time com **Algoritmo Genético**
        4. Visualize as **predições** e o **time otimizado** nas abas seguintes
        """)
    with col_b:
        st.markdown("#### ⚡ Features Táticas")
        st.markdown("""
        | Condição | Boost |
        |---|---|
        | ATA vs defesa fraca | **+30%** |
        | MEI vs defesa fraca | **+25%** |
        | ZAG em casa vitorioso | **+15%** |
        | Qualquer posição em casa | **+10%** |
        | Alta variância | **penalidade** |
        | 4+ do mesmo clube | **penalidade severa** |
        """)

    st.markdown("<hr class='custom'>", unsafe_allow_html=True)
    st.markdown("#### 📋 Pipeline de Execução")
    steps = [
        ("📡", "Coleta API", "Mercado, atletas prováveis, partidas da rodada"),
        ("🔧", "Feature Engineering", "18+ features: rolling, EWM, mando, força adversário, momento"),
        ("🧠", "Treinamento ML", "Random Forest / Gradient Boosting com validação temporal"),
        ("🎯", "Pesos Táticos", "Multiplicadores por posição + contexto de partida"),
        ("🧬", "Algoritmo Genético", f"Pop {population_size}, {generations} gerações, max {max_mesmo_clube}/clube"),
        ("💾", "Resultado", "Time em CSV + métricas de confiança"),
    ]
    step_cols = st.columns(len(steps))
    for (icon, title, desc), col in zip(steps, step_cols):
        with col:
            st.markdown(f"""
            <div style="background:#161b22; border:1px solid #30363d; border-radius:10px;
                        padding:16px; text-align:center; height:140px;">
                <div style="font-size:1.8rem;">{icon}</div>
                <div style="color:#e6edf3; font-weight:600; font-size:0.85rem; margin:6px 0 4px;">{title}</div>
                <div style="color:#8b949e; font-size:0.72rem; line-height:1.4;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# ABA ANÁLISE DA RODADA
# ══════════════════════════════════════════════════════════════
with tab_analise:

    st.markdown("#### ▶ Executar Pipeline Completo")

    col_btn, col_status = st.columns([1, 3])
    with col_btn:
        executar = st.button("🚀 Executar Análise", use_container_width=True, type="primary")

    if executar:
        with st.spinner("Inicializando pipeline..."):

            progress = st.progress(0, text="🔄 Preparando...")
            log_area = st.empty()
            logs = []

            def log(msg, tipo="info"):
                emoji = {"info":"ℹ️","ok":"✅","warn":"⚠️","err":"❌"}.get(tipo,"ℹ️")
                ts = datetime.now().strftime("%H:%M:%S")
                logs.append(f"`{ts}` {emoji} {msg}")
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

                collector.collect_partidas(rodada_atual)
                progress.progress(35, "📚 Carregando histórico...")

                # ── 3. Histórico ───────────────────────────────
                conn = sqlite3.connect(str(collector.db_path))
                historico_df = pd.read_sql_query("""
                    SELECT p.*, a.clube_id FROM pontuacoes p
                    JOIN atletas a ON p.atleta_id = a.atleta_id
                    WHERE p.rodada >= ? AND p.rodada <= ?
                    ORDER BY p.atleta_id, p.rodada
                """, conn, params=(max(1, rodada_atual - 15), rodada_atual))

                partidas_df = pd.read_sql_query("""
                    SELECT * FROM partidas WHERE rodada >= ? AND rodada <= ?
                """, conn, params=(max(1, rodada_atual - 15), rodada_atual))
                conn.close()

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
                from src.ml.predictor import CartolaPredictor
                predictor = CartolaPredictor(model_type=model_key)

                dados_treino = historico_df[historico_df['rodada'] < rodada_atual] if len(historico_df) > 0 else pd.DataFrame()
                can_train    = len(dados_treino) >= CartolaPredictor.HISTORICO_MINIMO

                if can_train:
                    metrics = predictor.train(dados_treino, validate=True)
                    log(f"Modelo treinado — MAE: {metrics['mae']:.2f} | R²: {metrics['r2']:.3f}", "ok")
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
                else:
                    predicoes_df = CartolaPredictor.fallback_heuristica(atletas_df)
                    predicoes_df['predicao_ajustada'] = predicoes_df['predicao']

                log(f"Predições geradas para {len(predicoes_df)} atletas", "ok")
                progress.progress(82, "🧬 Otimizando time genético...")

                # ── 7. Otimizador ──────────────────────────────
                from src.ml.optimizer import GeneticTeamOptimizer

                predicoes_opt = predicoes_df.copy()
                if 'predicao_std' not in predicoes_opt.columns:
                    predicoes_opt['predicao_std'] = 0.0
                score_col = 'predicao_ajustada' if 'predicao_ajustada' in predicoes_opt.columns else 'predicao'
                predicoes_opt['predicao'] = predicoes_opt[score_col]

                optimizer = GeneticTeamOptimizer(
                    atletas_df=atletas_df,
                    predicoes=predicoes_opt,
                    patrimonio=PATRIMONIO,
                    formacao=FORMACAO,
                    population_size=population_size,
                    generations=generations,
                    mutation_rate=mutation_rate,
                    elite_size=max(10, generations // 8),
                    max_mesmo_clube=max_mesmo_clube,
                    penalidade_variancia=True,
                )

                best_team, stats = optimizer.optimize()
                team_df = optimizer.format_team_output(best_team)
                log(f"Time otimizado: {stats['total_pontos_preditos']:.1f} pts | C$ {stats['total_preco']:.1f}", "ok")

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

            except Exception as e:
                import traceback
                st.markdown(f'<div class="alert-error">❌ Erro: {e}</div>', unsafe_allow_html=True)
                with st.expander("🔍 Detalhes do erro"):
                    st.code(traceback.format_exc())

    # ── Resultado da análise (persiste por sessão) ─────────
    if st.session_state.get('executado'):
        predicoes_df = st.session_state['predicoes_df']
        rodada_atual  = st.session_state['rodada']

        st.markdown("<hr class='custom'>", unsafe_allow_html=True)
        st.markdown(f"#### 📊 Predições — Rodada {rodada_atual}")

        score_col = 'predicao_ajustada' if 'predicao_ajustada' in predicoes_df.columns else 'predicao'

        # Filtros
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            pos_filter = st.multiselect(
                "Posição", options=[1,2,3,4,5,6],
                format_func=lambda x: POSICAO_NOME[x],
                default=[1,2,3,4,5,6]
            )
        with col_f2:
            top_n = st.slider("Top N atletas", 5, 50, 20)
        with col_f3:
            preco_max = st.number_input("Preço máximo (C$)", min_value=0.0, max_value=50.0, value=50.0, step=1.0)

        # Filtrar e exibir
        pred_filtrado = predicoes_df[predicoes_df['posicao_id'].isin(pos_filter)]
        if preco_max < 50.0:
            pred_filtrado = pred_filtrado[pred_filtrado['preco'] <= preco_max]
        pred_filtrado = pred_filtrado.nlargest(top_n, score_col).copy()

        # Montar tabela visual
        rows_html = ""
        for _, row in pred_filtrado.iterrows():
            pos_id  = int(row.get('posicao_id', 0))
            nome    = row.get('apelido', '—')
            preco   = row.get('preco', 0)
            pred_b  = row.get('predicao', 0)
            pred_aj = row.get(score_col, pred_b)
            boost   = (pred_aj / pred_b - 1) * 100 if pred_b > 0 else 0

            boost_tag = ""
            if boost > 1:
                boost_tag = f'<span class="boost-tag">+{boost:.0f}% tático</span>'

            rows_html += f"""
            <tr style="border-bottom:1px solid #21262d;">
                <td style="padding:10px 8px;">{badge_pos(pos_id)}</td>
                <td style="padding:10px 8px; font-weight:500; color:#e6edf3;">{nome}</td>
                <td style="padding:10px 8px; color:#8b949e;">{fmt_preco(preco)}</td>
                <td style="padding:10px 8px; color:#58a6ff;">{pred_b:.2f}</td>
                <td style="padding:10px 8px; color:#39d353; font-weight:600;">{pred_aj:.2f} {boost_tag}</td>
            </tr>"""

        st.markdown(f"""
        <div style="background:#161b22; border:1px solid #30363d; border-radius:12px; overflow:hidden;">
            <table style="width:100%; border-collapse:collapse; font-size:0.88rem;">
                <thead>
                    <tr style="background:#21262d; color:#8b949e; font-size:0.78rem; text-transform:uppercase; letter-spacing:.05em;">
                        <th style="padding:10px 8px; text-align:left;">Pos</th>
                        <th style="padding:10px 8px; text-align:left;">Atleta</th>
                        <th style="padding:10px 8px; text-align:left;">Preço</th>
                        <th style="padding:10px 8px; text-align:left;">Pred. Base</th>
                        <th style="padding:10px 8px; text-align:left;">Pred. Ajustada</th>
                    </tr>
                </thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)

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

        POS_ORDER = {1:1, 2:3, 3:2, 4:4, 5:5, 6:6}  # GOL, ZAG, LAT, MEI, ATA, TEC
        pos_order_rev = {1:'⛔ Goleiro', 2:'🛡️ Laterais', 3:'🛡️ Zagueiros',
                         4:'⚙️ Meias', 5:'🔥 Atacantes', 6:'📋 Técnico'}

        atletas_sorted = sorted(best_team, key=lambda a: POS_ORDER.get(a.get('posicao_id', 9), 9))

        current_pos = None
        for atleta in atletas_sorted:
            pos_id  = atleta.get('posicao_id', 0)
            if pos_id != current_pos:
                current_pos = pos_id
                st.markdown(f"**{pos_order_rev.get(pos_id, '?')}**")

            nome    = atleta.get('apelido', atleta.get('nome', '—'))
            preco   = atleta.get('preco', 0)
            pred    = atleta.get('predicao', 0)
            media   = atleta.get('media', 0)
            boost   = atleta.get('bonus_oponente_fraco', 0)
            em_casa = atleta.get('mando_casa', 0)

            boost_tags = ""
            if boost > 0.7 and pos_id in [4, 5]:
                boost_tags += '<span class="boost-tag">🎯 Defesa fraca</span> '
            if em_casa:
                boost_tags += '<span class="boost-tag">🏠 Casa</span> '
            if pred > 8:
                boost_tags += '<span class="boost-tag">⭐ Elite</span> '

            pct_bar = min(100, (pred / 12) * 100)
            col_nome, col_preco, col_pred, col_extra = st.columns([3, 1.2, 2.5, 2])
            with col_nome:
                st.markdown(
                    f"{badge_pos(pos_id)} &nbsp; **{nome}**",
                    unsafe_allow_html=True
                )
            with col_preco:
                st.markdown(f"<span style='color:#8b949e'>{fmt_preco(preco)}</span>", unsafe_allow_html=True)
            with col_pred:
                st.markdown(f"""
                <div style="display:flex; align-items:center; gap:8px;">
                    <div class="progress-bar-bg" style="flex:1">
                        <div class="progress-bar-fill" style="width:{pct_bar:.0f}%"></div>
                    </div>
                    <span style="color:#e6edf3; font-weight:600; min-width:42px;">{pred:.1f}pts</span>
                </div>""", unsafe_allow_html=True)
            with col_extra:
                st.markdown(boost_tags or "", unsafe_allow_html=True)

        # ── Comparativo com médias ─────────────────────────
        st.markdown("<hr class='custom'>", unsafe_allow_html=True)
        st.markdown("#### 📊 Distribuição por Posição")

        rows_pos = []
        pos_groups = {}
        for a in best_team:
            pid = a.get('posicao_id', 0)
            pos_groups.setdefault(pid, []).append(a)

        for pid, atletas in sorted(pos_groups.items()):
            nomes  = ", ".join(a.get('apelido', '?') for a in atletas)
            total  = sum(a.get('predicao', 0) for a in atletas)
            custo  = sum(a.get('preco', 0) for a in atletas)
            rows_pos.append({
                'Posição': POSICAO_NOME.get(pid, '?'),
                'Atletas': nomes,
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
