#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
========================================================================
SELEÇÃO DA RODADA
========================================================================
Página principal de escalação otimizada para a rodada atual.
Inspirада na funcionalidade 'Seleção da Rodada' da Máquina do Cartola.

Funcionalidades:
  - Carrega jogadores disponíveis via API do Cartola
  - Executa Feature Engineering V2 + MegaStrategy (PuLP)
  - Exibe campo visual com a escalação recomendada
  - Suporte a Reserva de Luxo
  - Filtros por formação e orçamento
========================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.feature_engineering_v2 import FeatureEngineeringV2
from src.optimizer.factory import CartolaOptimizer

st.set_page_config(
    page_title="Seleção da Rodada | Cartola AI",
    page_icon="⚽",
    layout="wide",
)

# ── Estilo ───────────────────────────────────────────────────────────
st.markdown("""
<style>
.campo-card {
    background: linear-gradient(135deg, #1a6b1a 0%, #2d8a2d 50%, #1a6b1a 100%);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    color: white;
    font-weight: bold;
    margin: 4px;
    border: 2px solid rgba(255,255,255,0.2);
    font-size: 12px;
    min-height: 80px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
}
.campo-card .nome { font-size: 13px; font-weight: 700; }
.campo-card .score { font-size: 16px; color: #FFD700; }
.campo-card .preco { font-size: 11px; color: #a8f0a8; }
.pos-badge {
    background: rgba(0,0,0,0.4);
    border-radius: 4px;
    padding: 1px 6px;
    font-size: 10px;
    margin-bottom: 4px;
}
style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────
st.title("⚽ Seleção da Rodada")
st.markdown("Escalação otimizada com **Programação Linear (PuLP)** + **Feature Engineering V2**")

# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configurações")
    budget = st.slider("💰 Orçamento (C$)", 80.0, 130.0, 100.0, 0.5)
    formacao = st.selectbox(
        "⚽ Formação",
        ["Auto (melhor)", "3-4-3", "3-5-2", "4-3-3", "4-4-2", "4-5-1", "5-3-2", "5-4-1"],
    )
    usar_reserva = st.toggle("🔄 Reserva de Luxo", value=True)
    max_por_clube = st.slider("🏟️ Máx. jogadores por clube", 1, 5, 3)
    st.divider()
    st.caption("ℹ️ Os dados são carregados do arquivo de jogadores disponível na pasta /data")

# ── Carregamento de dados ─────────────────────────────────────────────
@st.cache_data(ttl=300)
def carregar_jogadores():
    """Tenta carregar jogadores reais; gera sintéticos como fallback."""
    import glob
    csvs = glob.glob("data/jogadores*.csv") + glob.glob("data/atletas*.csv")
    if csvs:
        df = pd.read_csv(csvs[0])
        return df, "real"
    # Dados sintéticos para demo
    rng = np.random.default_rng(42)
    n = 120
    posicoes = ([1]*12 + [2]*20 + [3]*20 + [4]*36 + [5]*24 + [6]*8)
    df = pd.DataFrame({
        'atleta_id': range(1, n+1),
        'apelido': [f"Jogador_{i}" for i in range(1, n+1)],
        'posicao_id': posicoes[:n],
        'clube_id': rng.integers(1, 21, n),
        'preco': rng.uniform(4.0, 35.0, n).round(2),
        'media': rng.uniform(2.0, 10.0, n).round(2),
        'pontos_ultimas_5': rng.uniform(1.0, 12.0, n).round(2),
        'jogos': rng.integers(3, 20, n),
        'minutos_jogados': rng.integers(200, 1600, n),
        'status': rng.choice(['Provável', 'Provável', 'Provável', 'Dúvida', 'Suspenso'], n),
    })
    return df, "sintetico"

df_raw, fonte = carregar_jogadores()

if fonte == "sintetico":
    st.info("📊 Usando dados sintéticos para demonstração. Adicione `data/jogadores.csv` com dados reais da API do Cartola.")

# ── Feature Engineering ───────────────────────────────────────────────
with st.spinner("🧠 Calculando features e mega_score..."):
    fe = FeatureEngineeringV2()
    try:
        df_feat = fe.engineer_features(df_raw)
    except Exception as e:
        st.error(f"Erro no Feature Engineering: {e}")
        st.stop()

# ── Botão de otimizar ─────────────────────────────────────────────────
col_btn, col_info = st.columns([1, 3])
with col_btn:
    otimizar = st.button("🚀 Gerar Escalação", type="primary", use_container_width=True)
with col_info:
    st.metric("Jogadores disponíveis", len(df_feat))

if otimizar or 'lineup_result' in st.session_state:
    if otimizar:
        with st.spinner("⚙️ Otimizando escalação..."):
            try:
                optimizer = CartolaOptimizer(
                    strategy='mega',
                    config={'max_players_per_club': max_por_clube, 'solver_time_limit': 30}
                )
                form_arg = None if formacao == "Auto (melhor)" else formacao

                if usar_reserva:
                    result = optimizer.strategy.optimize_with_luxury_reserve(
                        df_feat, budget=budget, formation=form_arg
                    )
                    if result:
                        st.session_state['lineup_result'] = result
                        st.session_state['modo_reserva'] = True
                    else:
                        st.error("❌ Não foi possível gerar escalação com os parâmetros escolhidos.")
                        st.stop()
                else:
                    lineup = optimizer.optimize(df_feat, budget=budget, formation=form_arg)
                    if lineup is not None:
                        st.session_state['lineup_result'] = {'titulares': lineup, 'reservas': pd.DataFrame()}
                        st.session_state['modo_reserva'] = False
                    else:
                        st.error("❌ Não foi possível gerar escalação. Tente aumentar o orçamento ou reduzir restrições.")
                        st.stop()
            except Exception as e:
                st.error(f"Erro na otimização: {e}")
                st.stop()

    result = st.session_state.get('lineup_result', {})
    titulares = result.get('titulares', pd.DataFrame())
    reservas = result.get('reservas', pd.DataFrame())
    modo_reserva = st.session_state.get('modo_reserva', False)

    if titulares.empty:
        st.warning("Nenhuma escalação gerada ainda.")
    else:
        # ── Métricas top ──────────────────────────────────────────────
        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        custo = titulares['preco'].sum()
        score = titulares['mega_score'].sum() if 'mega_score' in titulares.columns else 0
        custo_total = result.get('custo_total', custo)
        m1.metric("💰 Custo Titulares", f"C$ {custo:.1f}")
        m2.metric("💰 Custo Total", f"C$ {custo_total:.1f}", delta=f"de C$ {budget:.0f}")
        m3.metric("🎯 Mega Score", f"{score:.1f}")
        m4.metric("👥 Jogadores", len(titulares))

        # ── Campo visual ──────────────────────────────────────────────
        st.subheader("🟢 Campo — Escalação Otimizada")

        POS_NOME = {1: 'GOL', 2: 'LAT', 3: 'ZAG', 4: 'MEI', 5: 'ATA', 6: 'TEC'}
        POS_ORDEM = [5, 4, 3, 2, 1, 6]  # ATA → TEC (de cima para baixo no campo)

        for pos_id in POS_ORDEM:
            jogadores_pos = titulares[titulares['posicao_id'] == pos_id]
            if jogadores_pos.empty:
                continue
            pos_nome = POS_NOME.get(pos_id, '?')
            cols = st.columns(len(jogadores_pos))
            for col, (_, row) in zip(cols, jogadores_pos.iterrows()):
                nome = row.get('apelido', row.get('nome', 'N/A'))
                mega = row.get('mega_score', 0)
                preco = row.get('preco', 0)
                col.markdown(f"""
                <div class='campo-card'>
                    <div class='pos-badge'>{pos_nome}</div>
                    <div class='nome'>{nome}</div>
                    <div class='score'>⭐ {mega:.1f}</div>
                    <div class='preco'>C$ {preco:.1f}</div>
                </div>
                """, unsafe_allow_html=True)

        # ── Reservas ──────────────────────────────────────────────────
        if modo_reserva and not reservas.empty:
            st.subheader("🔄 Reservas de Luxo")
            res_cols = st.columns(len(reservas))
            for col, (_, row) in zip(res_cols, reservas.iterrows()):
                nome = row.get('apelido', row.get('nome', 'N/A'))
                pos_nome = POS_NOME.get(int(row.get('posicao_id', 0)), '?')
                ev = row.get('ev_reserva', 0)
                preco = row.get('preco', 0)
                col.markdown(f"""
                <div class='campo-card' style='background:linear-gradient(135deg,#1a3a6b,#2d5a8a);'>
                    <div class='pos-badge'>{pos_nome}</div>
                    <div class='nome'>{nome}</div>
                    <div class='score'>EV +{ev:.1f}</div>
                    <div class='preco'>C$ {preco:.1f}</div>
                </div>
                """, unsafe_allow_html=True)

        # ── Tabela detalhada ──────────────────────────────────────────
        st.subheader("📋 Detalhes da Escalação")
        cols_show = [c for c in ['apelido', 'posicao_id', 'clube_id', 'mega_score', 'media', 'pontos_ultimas_5', 'preco', 'status'] if c in titulares.columns]
        st.dataframe(
            titulares[cols_show].sort_values('posicao_id').reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )

        # ── Download ──────────────────────────────────────────────────
        csv = titulares.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Baixar escalação CSV", csv, "escalacao_otimizada.csv", "text/csv")
