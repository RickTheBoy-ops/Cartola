import pandas as pd
import numpy as np
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Criação de features avançadas para predição do Cartola FC.
    Inclui:
    - Médias móveis simples e ponderadas
    - Tendência de forma
    - Força do adversário e mando de campo
    - Ratios de scouts por posição
    - Features de preço e valorização
    - Regularidade (coeficiente de variação)
    - Momento do clube (vitórias recentes)
    - Força defensiva do adversário
    - Ponderação por posição
    """

    # Pesos dos scouts por posição (quanto cada scout impacta a pontuação real)
    SCOUT_WEIGHTS_POR_POSICAO = {
        1: {'DE': 1.0, 'DP': 7.0, 'SG': 5.0, 'GS': -1.0, 'GC': -3.0, 'CV': -3.0, 'CA': -1.0},  # Goleiro
        2: {'DS': 1.2, 'SG': 5.0, 'A': 5.0, 'FS': 0.5, 'FC': -0.3, 'CA': -1.0, 'CV': -3.0, 'G': 8.0},  # Lateral
        3: {'SG': 5.0, 'DS': 1.2, 'I': 0.5, 'FC': -0.3, 'CA': -1.0, 'CV': -3.0, 'G': 8.0, 'GC': -3.0},  # Zagueiro
        4: {'G': 8.0, 'A': 5.0, 'FT': 3.0, 'FD': 1.0, 'DS': 1.2, 'FS': 0.5, 'CA': -1.0, 'CV': -3.0},  # Meia
        5: {'G': 8.0, 'A': 5.0, 'FT': 3.0, 'FD': 1.0, 'FF': 0.8, 'PE': 1.0, 'CA': -1.0, 'CV': -3.0},  # Atacante
        6: {},  # Técnico (sem scouts individuais)
    }

    @staticmethod
    def create_rolling_features(df: pd.DataFrame, windows: List[int] = [3, 5, 8]) -> pd.DataFrame:
        """
        Cria médias móveis simples e ponderadas (peso exponencial).
        Janela [3, 5, 8] para capturar forma recente, média e tendência longa.
        """
        df = df.sort_values(['atleta_id', 'rodada'])

        for window in windows:
            # Média móvel simples
            df[f'pontos_media_{window}'] = df.groupby('atleta_id')['pontos'].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )

            # Desvio padrão (consistência)
            df[f'pontos_std_{window}'] = df.groupby('atleta_id')['pontos'].transform(
                lambda x: x.rolling(window, min_periods=1).std()
            )

            # Máximo recente
            df[f'pontos_max_{window}'] = df.groupby('atleta_id')['pontos'].transform(
                lambda x: x.rolling(window, min_periods=1).max()
            )

            # Mínimo recente
            df[f'pontos_min_{window}'] = df.groupby('atleta_id')['pontos'].transform(
                lambda x: x.rolling(window, min_periods=1).min()
            )

        # Média Móvel Ponderada Exponencial (mais peso para jogos recentes)
        df['pontos_ewm_3'] = df.groupby('atleta_id')['pontos'].transform(
            lambda x: x.ewm(span=3, min_periods=1).mean()
        )
        df['pontos_ewm_5'] = df.groupby('atleta_id')['pontos'].transform(
            lambda x: x.ewm(span=5, min_periods=1).mean()
        )

        return df

    @staticmethod
    def create_form_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features de 'momento' do jogador.
        - Tendência (média curta - média longa)
        - Sequência positiva
        - Pontos por 90 minutos
        - Regularidade (coeficiente de variação)
        """
        df = df.sort_values(['atleta_id', 'rodada'])

        # Tendência (últimas 3 rodadas vs últimas 8 rodadas)
        media_curta = df.groupby('atleta_id')['pontos'].transform(
            lambda x: x.rolling(3, min_periods=1).mean()
        )
        media_longa = df.groupby('atleta_id')['pontos'].transform(
            lambda x: x.rolling(8, min_periods=1).mean()
        )
        df['tendencia'] = media_curta - media_longa

        # Contagem de rodadas consecutivas pontuando acima da média
        def safe_sequencia(group):
            try:
                if 'media' not in group.columns or len(group) == 0:
                    return pd.Series(0, index=group.index)
                mask = group['pontos'] > group['media']
                return mask.groupby((~mask).cumsum()).cumsum()
            except Exception:
                return pd.Series(0, index=group.index)

        try:
            df['sequencia_positiva'] = df.groupby('atleta_id').apply(safe_sequencia).reset_index(level=0, drop=True)
        except Exception:
            df['sequencia_positiva'] = 0

        # Pontos por 90 minutos (eficiência)
        minutos = df.get('minutos_jogados', pd.Series(0, index=df.index))
        df['pontos_por_90min'] = np.where(
            minutos > 0,
            (df['pontos'] / minutos) * 90,
            0
        )

        # Regularidade (coeficiente de variação = std / média)
        # Quanto menor o CV, mais regular o jogador
        df['regularidade'] = np.where(
            df.get('pontos_media_5', df.get('media', pd.Series(1, index=df.index))) > 0,
            df.get('pontos_std_5', pd.Series(0, index=df.index)) /
            df.get('pontos_media_5', df.get('media', pd.Series(1, index=df.index))),
            1.0
        )

        return df

    @staticmethod
    def create_opponent_features(df: pd.DataFrame, partidas_df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria features relacionadas ao adversário e mando de campo.
        Tratamento robusto para dados ausentes.
        """
        if partidas_df is None or len(partidas_df) == 0:
            df['forca_adversario'] = 0.5
            df['mando_casa'] = 1
            return df

        if 'clube_id' not in df.columns:
            df['forca_adversario'] = 0.5
            df['mando_casa'] = 1
            return df

        # Merge como mandante
        df_merged = df.merge(
            partidas_df[['rodada', 'clube_casa_id', 'clube_visitante_id',
                          'aproveitamento_mandante', 'aproveitamento_visitante']],
            left_on=['rodada', 'clube_id'],
            right_on=['rodada', 'clube_casa_id'],
            how='left'
        )

        # Identificar quem é mandante e visitante
        eh_casa = df_merged['clube_casa_id'].notna()

        # Para os que não deram match como mandante, tentar como visitante
        sem_match = ~eh_casa
        if sem_match.any():
            merge_visit = df.loc[sem_match.values].merge(
                partidas_df[['rodada', 'clube_casa_id', 'clube_visitante_id',
                              'aproveitamento_mandante', 'aproveitamento_visitante']],
                left_on=['rodada', 'clube_id'],
                right_on=['rodada', 'clube_visitante_id'],
                how='left'
            )

            # Substituir linhas sem match
            for col in ['clube_casa_id', 'clube_visitante_id', 'aproveitamento_mandante', 'aproveitamento_visitante']:
                if col in merge_visit.columns:
                    df_merged.loc[sem_match, col] = merge_visit[col].values

        # Recalcular eh_casa após merge completo
        eh_casa = (df_merged['clube_id'] == df_merged['clube_casa_id'])

        # Força do adversário
        df_merged['forca_adversario'] = np.where(
            eh_casa,
            df_merged['aproveitamento_visitante'].fillna(0.5),
            df_merged['aproveitamento_mandante'].fillna(0.5)
        )

        # Mando de campo
        df_merged['mando_casa'] = eh_casa.astype(int)

        # Remover colunas temporárias de merge
        cols_remover = ['clube_casa_id', 'clube_visitante_id', 'aproveitamento_mandante', 'aproveitamento_visitante']
        df_merged = df_merged.drop(columns=[c for c in cols_remover if c in df_merged.columns], errors='ignore')

        return df_merged

    @staticmethod
    def create_scout_ratios(df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria ratios e proporções de scouts
        """
        # Verificar se colunas de scouts existem
        scouts_cols = ['G', 'FD', 'FT', 'FF', 'A', 'FS', 'DS', 'FC', 'DE']
        for col in scouts_cols:
            if col not in df.columns:
                df[col] = 0

        # Gols por finalização (eficiência ofensiva)
        total_finalizacoes = df['FD'] + df['FT'] + df['FF']
        df['gols_por_finalizacao'] = np.where(
            total_finalizacoes > 0,
            df['G'] / total_finalizacoes,
            0
        )

        # Taxa de participação em gols (G + A)
        df['participacao_gols'] = df.get('G', 0) + df.get('A', 0)

        # Eficiência defensiva (desarmes por falta cometida)
        df['eficiencia_defensiva'] = np.where(
            df['FC'] > 0,
            df['DS'] / df['FC'],
            df['DS']
        )

        # Score defensivo (defesas + desarmes)
        df['score_defensivo'] = df['DE'] + df['DS']

        return df

    @staticmethod
    def create_price_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Features relacionadas a preço e valorização
        """
        df = df.sort_values(['atleta_id', 'rodada'])

        # Variação de preço acumulada (últimas 5 rodadas)
        if 'variacao' in df.columns:
            df['variacao_acumulada_5'] = df.groupby('atleta_id')['variacao'].transform(
                lambda x: x.rolling(5, min_periods=1).sum()
            )
        else:
            df['variacao_acumulada_5'] = 0

        # Relação preço/pontos (custo-benefício)
        preco = df.get('preco', pd.Series(1, index=df.index))
        media = df.get('media', pd.Series(0, index=df.index))
        df['custo_beneficio'] = np.where(
            preco > 0,
            media / preco,
            0
        )

        # Mínima Pontuação para Valorizar (MPV)
        df['mpv'] = preco * 0.02
        df['distancia_mpv'] = media - df['mpv']

        return df

    @classmethod
    def create_position_score(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cria score baseado nos scouts mais relevantes para cada posição.
        Aplica pesos diferenciados por posição.
        """
        if 'posicao_id' not in df.columns:
            df['position_score'] = 0
            return df

        df['position_score'] = 0.0

        for pos_id, weights in cls.SCOUT_WEIGHTS_POR_POSICAO.items():
            mask = df['posicao_id'] == pos_id
            if not mask.any() or not weights:
                continue

            score = pd.Series(0.0, index=df.index)
            for scout, weight in weights.items():
                if scout in df.columns:
                    score += df[scout].fillna(0) * weight

            df.loc[mask, 'position_score'] = score[mask]

        return df

    # ---------------------------------------------------------------
    # NOVAS FEATURES TÁTICAS
    # ---------------------------------------------------------------

    @staticmethod
    def add_club_momentum_features(df: pd.DataFrame, partidas_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula vitórias do clube nas últimas 3 rodadas (momento).
        Produz a feature 'clube_vitorias_recentes' (int 0-3).
        """
        if partidas_df is None or len(partidas_df) == 0:
            df['clube_vitorias_recentes'] = 0
            return df

        cols_necessarias = {'rodada', 'clube_casa_id', 'clube_visitante_id',
                            'placar_oficial_mandante', 'placar_oficial_visitante'}
        if not cols_necessarias.issubset(partidas_df.columns):
            df['clube_vitorias_recentes'] = 0
            return df

        try:
            # Expansão de partidas: uma linha por clube por jogo
            mandante = partidas_df[['rodada', 'clube_casa_id', 'placar_oficial_mandante', 'placar_oficial_visitante']].copy()
            mandante.columns = ['rodada', 'clube_id', 'gols_pro', 'gols_contra']
            mandante['vitoria'] = (mandante['gols_pro'] > mandante['gols_contra']).astype(int)

            visitante = partidas_df[['rodada', 'clube_visitante_id', 'placar_oficial_visitante', 'placar_oficial_mandante']].copy()
            visitante.columns = ['rodada', 'clube_id', 'gols_pro', 'gols_contra']
            visitante['vitoria'] = (visitante['gols_pro'] > visitante['gols_contra']).astype(int)

            resultados = pd.concat([mandante, visitante], ignore_index=True)
            resultados = resultados.sort_values('rodada')

            # Vitórias nas últimas 3 rodadas por clube
            momento = (
                resultados
                .groupby('clube_id')['vitoria']
                .apply(lambda x: x.rolling(3, min_periods=1).sum().iloc[-1] if len(x) >= 1 else 0)
                .rename('clube_vitorias_recentes')
                .reset_index()
            )

            if 'clube_id' not in df.columns:
                df['clube_vitorias_recentes'] = 0
                return df

            df = df.merge(momento, on='clube_id', how='left')
            df['clube_vitorias_recentes'] = df['clube_vitorias_recentes'].fillna(0).astype(int)

        except Exception as e:
            logger.warning(f"⚠️ Erro em add_club_momentum_features: {e}")
            df['clube_vitorias_recentes'] = 0

        return df

    @staticmethod
    def add_opponent_strength_features(df: pd.DataFrame, partidas_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula força defensiva do adversário.
        'bonus_oponente_fraco' → quanto maior, mais fraca a defesa inimiga
        (mais oportunidade para atacantes/meias pontuarem).
        """
        if partidas_df is None or len(partidas_df) == 0:
            df['bonus_oponente_fraco'] = 0.5
            return df

        cols_necessarias = {'clube_casa_id', 'clube_visitante_id',
                            'placar_oficial_mandante', 'placar_oficial_visitante'}
        if not cols_necessarias.issubset(partidas_df.columns):
            df['bonus_oponente_fraco'] = 0.5
            return df

        try:
            # Gols sofridos em casa
            gols_sofridos_casa = (
                partidas_df.groupby('clube_casa_id')['placar_oficial_visitante']
                .mean()
                .rename('gols_sofridos')
            )
            # Gols sofridos fora
            gols_sofridos_fora = (
                partidas_df.groupby('clube_visitante_id')['placar_oficial_mandante']
                .mean()
                .rename('gols_sofridos')
            )

            # Média combinada por clube
            defesa_por_clube = (
                pd.concat([gols_sofridos_casa, gols_sofridos_fora])
                .groupby(level=0)
                .mean()
                .rename('media_gols_sofridos')
                .reset_index()
                .rename(columns={'index': 'adversario_clube_id'})
            )
            defesa_por_clube.columns = ['adversario_clube_id', 'media_gols_sofridos']

            # Normaliza: bonus = gols_sofridos / (max + epsilon)
            max_gols = defesa_por_clube['media_gols_sofridos'].max()
            defesa_por_clube['bonus_oponente_fraco'] = (
                defesa_por_clube['media_gols_sofridos'] / (max_gols + 1e-6)
            ).clip(0, 1)

            # Identificar adversário de cada atleta via partidas
            if 'clube_id' not in df.columns:
                df['bonus_oponente_fraco'] = 0.5
                return df

            # Descobrir adversário por rodada+clube
            casa_map = partidas_df[['rodada', 'clube_casa_id', 'clube_visitante_id']].copy()
            casa_map.columns = ['rodada', 'clube_id', 'adversario_clube_id']

            visit_map = partidas_df[['rodada', 'clube_visitante_id', 'clube_casa_id']].copy()
            visit_map.columns = ['rodada', 'clube_id', 'adversario_clube_id']

            adversarios = pd.concat([casa_map, visit_map], ignore_index=True)

            df = df.merge(adversarios, on=['rodada', 'clube_id'], how='left')
            df = df.merge(defesa_por_clube[['adversario_clube_id', 'bonus_oponente_fraco']],
                          on='adversario_clube_id', how='left')
            df['bonus_oponente_fraco'] = df['bonus_oponente_fraco'].fillna(0.5)

            # Limpar coluna temporária
            df = df.drop(columns=['adversario_clube_id'], errors='ignore')

        except Exception as e:
            logger.warning(f"⚠️ Erro em add_opponent_strength_features: {e}")
            df['bonus_oponente_fraco'] = 0.5

        return df

    @staticmethod
    def add_position_weighted_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica pesos por posição para refletir variância natural:
        ATA (1.5) > MEI (1.2) > TEC (1.1) > LAT (1.0) > ZAG (0.9) > GOL (0.8)
        Produz 'peso_posicao' e 'pontos_ponderados'.
        """
        # 1=GOL, 2=LAT, 3=ZAG, 4=MEI, 5=ATA, 6=TEC
        POSICAO_PESO = {1: 0.8, 2: 1.0, 3: 0.9, 4: 1.2, 5: 1.5, 6: 1.1}

        if 'posicao_id' not in df.columns:
            df['peso_posicao'] = 1.0
            df['pontos_ponderados'] = df.get('pontos', 0)
            return df

        df['peso_posicao'] = df['posicao_id'].map(POSICAO_PESO).fillna(1.0)

        pontos_col = df.get('pontos', None)
        if pontos_col is not None:
            df['pontos_ponderados'] = df['pontos'] * df['peso_posicao']
        else:
            df['pontos_ponderados'] = 0.0

        return df

    # ---------------------------------------------------------------

    @classmethod
    def engineer_all_features(
        cls,
        df: pd.DataFrame,
        partidas_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Aplica todas as transformações de features.
        Ordem: rolling → form → opponent → scouts → price → position
               → momentum → oponent_strength → position_weights
        """
        logger.info(f"🔧 Iniciando feature engineering com {len(df)} registros...")

        df = cls.create_rolling_features(df)
        df = cls.create_form_features(df)
        df = cls.create_opponent_features(df, partidas_df)
        df = cls.create_scout_ratios(df)
        df = cls.create_price_features(df)
        df = cls.create_position_score(df)

        # === NOVAS FEATURES TÁTICAS ===
        df = cls.add_club_momentum_features(df, partidas_df)
        df = cls.add_opponent_strength_features(df, partidas_df)
        df = cls.add_position_weighted_features(df)

        # Remove NaN / infinitos
        df = df.fillna(0)
        df = df.replace([np.inf, -np.inf], 0)

        logger.info(f"✅ Feature engineering concluído: {df.shape[1]} colunas criadas")

        return df


    # ---------------------------------------------------------------
    # FEATURES MANDO DE CAMPO (MC/MF) E PONTOS CEDIDOS POR POSIÇÃO
    # ---------------------------------------------------------------

    @staticmethod
    def add_home_away_averages(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula médias separadas por mando de campo (MC e MF) para cada atleta.

        MC = média como mandante (últimas 3 e 5 rodadas jogando em casa)
        MF = média como visitante (últimas 3 e 5 rodadas jogando fora)

        Essas métricas são usadas pelos top cartoleiros para filtrar
        jogadores conforme o tipo de jogo da rodada.
        """
        if 'mando_casa' not in df.columns:
            df['mc_3'] = df.get('pontos_media_3', 0)
            df['mc_5'] = df.get('pontos_media_5', 0)
            df['mf_3'] = df.get('pontos_media_3', 0)
            df['mf_5'] = df.get('pontos_media_5', 0)
            return df

        df = df.sort_values(['atleta_id', 'rodada'])

        for window, suffix in [(3, '3'), (5, '5')]:
            mc_col = f'mc_{suffix}'
            mf_col = f'mf_{suffix}'

            def rolling_mean_home(group):
                home_pts = group.loc[group['mando_casa'] == 1, 'pontos']
                result = pd.Series(np.nan, index=group.index)
                # Preencher a média móvel apenas nas rodadas mandante
                rolling_vals = home_pts.rolling(window, min_periods=1).mean()
                result.loc[rolling_vals.index] = rolling_vals
                # Forward fill para que o valor fique disponível em rodadas visitante também
                result = result.ffill().bfill().fillna(0)
                return result

            def rolling_mean_away(group):
                away_pts = group.loc[group['mando_casa'] == 0, 'pontos']
                result = pd.Series(np.nan, index=group.index)
                rolling_vals = away_pts.rolling(window, min_periods=1).mean()
                result.loc[rolling_vals.index] = rolling_vals
                result = result.ffill().bfill().fillna(0)
                return result

            try:
                df[mc_col] = df.groupby('atleta_id', group_keys=False).apply(rolling_mean_home)
                df[mf_col] = df.groupby('atleta_id', group_keys=False).apply(rolling_mean_away)
            except Exception as e:
                logger.warning(f"⚠️ Erro em add_home_away_averages (window={window}): {e}")
                df[mc_col] = df.get('pontos_media_3', 0)
                df[mf_col] = df.get('pontos_media_3', 0)

        # Vantagem mandante: diferença mc_5 - mf_5
        df['vantagem_mandante'] = df['mc_5'] - df['mf_5']

        # Feature final: média relevante conforme mando da rodada atual
        df['media_mando_relevante'] = np.where(
            df['mando_casa'] == 1,
            df['mc_5'],
            df['mf_5']
        )

        logger.info("✅ Features MC/MF calculadas (janelas 3 e 5)")
        return df

    @staticmethod
    def add_points_conceded_by_position(df: pd.DataFrame, partidas_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula quantos pontos cada time cedeu para cada posição historicamente.

        'pontos_cedidos_posicao' = média de pontos que o adversário concede
        para a posição do atleta → quanto maior, mais fraca é a defesa
        do adversário naquela posição → mais favorável para escalar.

        Exemplo prático:
        - Lateral do Flamengo vai enfrentar Botafogo
        - Botafogo cedeu em média 7.2 pts para laterais nos últimos 5 jogos
        - Isso aumenta o score do lateral do Flamengo nessa rodada
        """
        if partidas_df is None or len(partidas_df) == 0:
            df['pontos_cedidos_posicao'] = 0.5
            return df

        if 'posicao_id' not in df.columns or 'clube_id' not in df.columns:
            df['pontos_cedidos_posicao'] = 0.5
            return df

        try:
            # Precisamos do histórico de pontos por atleta/partida para calcular
            # quanto cada time cedeu por posição
            # df deve ter: atleta_id, clube_id, posicao_id, pontos, rodada
            # partidas_df deve ter: clube_casa_id, clube_visitante_id, rodada

            # Mapeamento adversário por rodada e clube
            casa_map = partidas_df[['rodada', 'clube_casa_id', 'clube_visitante_id']].copy()
            casa_map.columns = ['rodada', 'clube_id', 'adversario_id']

            visit_map = partidas_df[['rodada', 'clube_visitante_id', 'clube_casa_id']].copy()
            visit_map.columns = ['rodada', 'clube_id', 'adversario_id']

            adversarios = pd.concat([casa_map, visit_map], ignore_index=True)

            # Join atleta -> adversário
            df_temp = df.merge(adversarios, on=['rodada', 'clube_id'], how='left')

            if 'adversario_id' not in df_temp.columns or df_temp['adversario_id'].isna().all():
                df['pontos_cedidos_posicao'] = 0.5
                return df

            # Pontos cedidos por time por posição (visão do adversário)
            # O adversário "cedeu" os pontos que o atleta fez contra ele
            cedidos = (
                df_temp
                .groupby(['adversario_id', 'posicao_id'])['pontos']
                .mean()
                .reset_index()
                .rename(columns={
                    'adversario_id': 'adversario_clube_id',
                    'pontos': 'pontos_cedidos_posicao'
                })
            )

            # Normalizar 0-1 dentro de cada posição
            for pos in cedidos['posicao_id'].unique():
                mask = cedidos['posicao_id'] == pos
                vals = cedidos.loc[mask, 'pontos_cedidos_posicao']
                max_val = vals.max()
                if max_val > 0:
                    cedidos.loc[mask, 'pontos_cedidos_posicao'] = (vals / max_val).clip(0, 1)

            # Adicionar adversário da rodada atual ao df original
            df = df.merge(adversarios, on=['rodada', 'clube_id'], how='left')
            df = df.rename(columns={'adversario_id': 'adversario_clube_id'})
            df = df.merge(cedidos, on=['adversario_clube_id', 'posicao_id'], how='left')
            df['pontos_cedidos_posicao'] = df['pontos_cedidos_posicao'].fillna(0.5)

            # Remover coluna temporária
            df = df.drop(columns=['adversario_clube_id'], errors='ignore')

            logger.info("✅ Feature 'pontos_cedidos_posicao' calculada com sucesso")

        except Exception as e:
            logger.warning(f"⚠️ Erro em add_points_conceded_by_position: {e}")
            df['pontos_cedidos_posicao'] = 0.5

        return df

    @classmethod
    def engineer_all_features_v3(
        cls,
        df: pd.DataFrame,
        partidas_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Versão V3 do pipeline completo de features.
        Adiciona MC/MF e pontos cedidos por posição ao pipeline existente.
        """
        # Roda pipeline base (V2)
        df = cls.engineer_all_features(df, partidas_df)

        # === NOVAS FEATURES V3 ===
        df = cls.add_home_away_averages(df)
        df = cls.add_points_conceded_by_position(df, partidas_df)

        # Remover NaN / infinitos residuais
        df = df.fillna(0)
        df = df.replace([np.inf, -np.inf], 0)

        logger.info(f"✅ Feature engineering V3 concluído: {df.shape[1]} colunas")
        return df
