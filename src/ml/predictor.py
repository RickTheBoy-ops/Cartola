import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from typing import List, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class CartolaPredictor:
    """
    Modelo de predição de pontuação usando ensemble.
    - Validação de histórico mínimo
    - Fallback inteligente por média ponderada
    - Predição com intervalo de confiança
    """

    HISTORICO_MINIMO = 30  # Mínimo de registros para treinar ML

    def __init__(self, model_type: str = 'rf'):
        self.model_type = model_type
        self.model = None
        self.feature_columns = None
        self.is_trained = False

        if model_type == 'rf':
            self.model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gb':
            self.model = GradientBoostingRegressor(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=10,
                random_state=42
            )

    def prepare_features(self, df: pd.DataFrame, is_training: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepara features para treinamento/predição"""
        feature_cols = [
            # Médias móveis simples
            'pontos_media_3', 'pontos_media_5', 'pontos_media_8',
            'pontos_std_3', 'pontos_std_5',
            'pontos_max_5', 'pontos_min_5',

            # Médias móveis exponenciais
            'pontos_ewm_3', 'pontos_ewm_5',

            # Forma
            'tendencia', 'sequencia_positiva', 'pontos_por_90min',
            'regularidade',

            # Adversário
            'forca_adversario', 'mando_casa',

            # Scouts
            'gols_por_finalizacao', 'participacao_gols',
            'eficiencia_defensiva', 'score_defensivo',

            # Posição
            'position_score',

            # Preço
            'custo_beneficio', 'distancia_mpv', 'variacao_acumulada_5',

            # Básicas
            'posicao_id', 'minutos_jogados', 'jogos', 'media', 'preco'
        ]

        # Filtrar colunas que existem
        available_cols = [col for col in feature_cols if col in df.columns]

        X = df[available_cols].copy()
        y = df['pontos'].copy() if 'pontos' in df.columns else None

        if is_training:
            self.feature_columns = available_cols

        return X, y

    def train(self, df: pd.DataFrame, validate: bool = True) -> Dict:
        """Treina modelo com validação temporal"""
        if len(df) < self.HISTORICO_MINIMO:
            logger.warning(
                f"⚠️ Histórico insuficiente ({len(df)} < {self.HISTORICO_MINIMO}). "
                f"Usando fallback de média ponderada."
            )
            return {'mae': 0, 'rmse': 0, 'r2': 0, 'fallback': True}

        X, y = self.prepare_features(df, is_training=True)

        if y is None:
            raise ValueError("Target 'pontos' not found in dataframe!")

        logger.info(f"🧠 Treinando modelo com {len(X)} amostras e {len(X.columns)} features")

        if validate and len(X) > 10:
            tscv = TimeSeriesSplit(n_splits=min(5, len(X) // 2))
            try:
                scores = cross_val_score(
                    self.model, X, y,
                    cv=tscv,
                    scoring='neg_mean_absolute_error',
                    n_jobs=-1
                )
                logger.info(f"📊 MAE Cross-Validation: {-scores.mean():.2f} (+/- {scores.std():.2f})")
            except Exception as e:
                logger.warning(f"⚠️ Erro no Cross-Validation: {e}")

        # Treinar no dataset completo
        self.model.fit(X, y)
        self.is_trained = True

        # Métricas de treino
        y_pred = self.model.predict(X)

        metrics = {
            'mae': mean_absolute_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'r2': r2_score(y, y_pred),
            'fallback': False
        }

        logger.info(
            f"📈 Métricas de treino - MAE: {metrics['mae']:.2f}, "
            f"RMSE: {metrics['rmse']:.2f}, R²: {metrics['r2']:.3f}"
        )

        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

            logger.info(f"\n🏆 Top 10 features:\n{feature_importance.head(10).to_string(index=False)}")

        return metrics

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Prediz pontuação para próxima rodada"""
        if not self.is_trained:
            raise ValueError("Modelo não foi treinado ainda!")

        X, _ = self.prepare_features(df)

        # Garantir mesmas features do treino
        missing = [c for c in self.feature_columns if c not in X.columns]
        for col in missing:
            X[col] = 0
        X = X[self.feature_columns]

        predictions = self.model.predict(X)

        # Não permitir predições negativas
        predictions = np.maximum(predictions, 0)

        return predictions

    def predict_with_confidence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prediz com intervalo de confiança (para Random Forest)"""
        predictions = self.predict(df)

        # Garantir que 'media' está presente
        cols_resultado = ['atleta_id', 'apelido', 'posicao_id', 'clube_id', 'preco']
        if 'media' in df.columns:
            cols_resultado.append('media')

        available = [c for c in cols_resultado if c in df.columns]
        result_df = df[available].copy()
        result_df['predicao'] = predictions

        if 'media' not in result_df.columns:
            result_df['media'] = 0

        # Se for Random Forest, calcular intervalo de confiança
        if self.model_type == 'rf' and hasattr(self.model, 'estimators_'):
            X, _ = self.prepare_features(df)
            missing = [c for c in self.feature_columns if c not in X.columns]
            for col in missing:
                X[col] = 0
            X = X[self.feature_columns]

            all_predictions = np.array([tree.predict(X) for tree in self.model.estimators_])

            result_df['predicao_std'] = all_predictions.std(axis=0)
            result_df['predicao_min'] = all_predictions.min(axis=0)
            result_df['predicao_max'] = all_predictions.max(axis=0)

        return result_df

    def predict_with_tactical_weights(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prediz com multiplicadores táticos por posição e contexto de partida.

        Multiplicadores aplicados:
        - ATA (pos 5) vs defesa fraca (bonus_oponente_fraco > 0.7)  → +30%
        - MEI (pos 4) vs defesa fraca (bonus_oponente_fraco > 0.6)  → +25%
        - ZAG (pos 3) em casa + clube vitorioso (>= 2 vitórias)    → +15%
        - Bnus geral em casa para todos                              → +10%
        """
        result_df = self.predict_with_confidence(df)

        # Inicializar multiplicador em 1.0
        multiplier = np.ones(len(result_df))

        # Reindexar df para alinhar com result_df (mesmo idx)
        df_align = df.copy().reset_index(drop=True)
        result_df = result_df.reset_index(drop=True)

        bonus = df_align.get('bonus_oponente_fraco', pd.Series(0.5, index=df_align.index))
        mando = df_align.get('mando_casa', pd.Series(0, index=df_align.index))
        momento = df_align.get('clube_vitorias_recentes', pd.Series(0, index=df_align.index))
        posicao = df_align.get('posicao_id', pd.Series(0, index=df_align.index))

        # Atacantes vs defesa fraca (+30%)
        mask_ata = (posicao == 5) & (bonus > 0.7)
        multiplier[mask_ata.values] *= 1.30

        # Meias vs defesa fraca (+25%)
        mask_mei = (posicao == 4) & (bonus > 0.6)
        multiplier[mask_mei.values] *= 1.25

        # Zagueiros em casa com time vitorioso (+15%)
        mask_zag = (posicao == 3) & (mando == 1) & (momento >= 2)
        multiplier[mask_zag.values] *= 1.15

        # Bônus universal: jogando em casa (+10%)
        mask_casa = mando == 1
        multiplier[mask_casa.values] *= 1.10

        result_df['predicao_ajustada'] = (result_df['predicao'] * multiplier).clip(lower=0)

        logger.info(
            f"🎯 Pesos táticos aplicados: ATA boost={mask_ata.sum()}, "
            f"MEI boost={mask_mei.sum()}, ZAG boost={mask_zag.sum()}, "
            f"Casa boost={mask_casa.sum()}"
        )

        return result_df

    @staticmethod
    def fallback_heuristica(atletas_df: pd.DataFrame) -> pd.DataFrame:
        """
        Fallback inteligente quando não há histórico suficiente para ML.
        Usa média ponderada: 60% média geral + 30% pontos recentes + 10% variação.
        """
        df = atletas_df.copy()

        cols_base = ['atleta_id', 'apelido', 'posicao_id', 'clube_id', 'preco']
        available = [c for c in cols_base if c in df.columns]
        result = df[available].copy()

        media = df.get('media', pd.Series(0, index=df.index))
        pontos = df.get('pontos', pd.Series(0, index=df.index))
        variacao = df.get('variacao', pd.Series(0, index=df.index))

        # Heurística: pontuação estimada combinando sinais disponíveis
        result['predicao'] = (media * 0.6) + (pontos * 0.3) + (variacao * 0.1)
        result['predicao'] = result['predicao'].clip(lower=0)
        result['media'] = media

        logger.info(f"🎯 Fallback heurístico gerado para {len(result)} atletas")

        return result

    def save_model(self, filepath: str):
        """Salva modelo treinado"""
        joblib.dump({
            'model': self.model,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type,
            'is_trained': self.is_trained
        }, filepath)
        logger.info(f"💾 Modelo salvo em: {filepath}")

    def load_model(self, filepath: str):
        """Carrega modelo treinado"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.feature_columns = data['feature_columns']
        self.model_type = data['model_type']
        self.is_trained = data.get('is_trained', True)
        logger.info(f"📂 Modelo carregado de: {filepath}")

    # ---------------------------------------------------------------
    # FLAG DE VALORIZAÇÃO
    # ---------------------------------------------------------------

    @staticmethod
    def add_valorizacao_flag(result_df: pd.DataFrame) -> pd.DataFrame:
        """
        Classifica cada jogador com selo de risco de valorização,
        similar ao sistema do Gato Mestre PRO.

        A fórmula de valorização do Cartola FC é:
            variacao = (pontos - MPV) * C
        onde MPV = preço_atual * 0.02 (aproximação do mínimo para valorizar)
        e C é uma constante que varia por faixa de preço.

        Selos:
        - 🟢 ÓTIMA ESCOLHA:  predicao >= MPV * 1.5  (boa margem de valorização)
        - 🟡 BOA ESCOLHA:    predicao >= MPV         (deve valorizar)
        - 🟠 ARRISCADO:      predicao >= MPV * 0.7   (pode empatar ou valorizar pouco)
        - 🔴 PODE ZICAR:     predicao <  MPV * 0.7   (alto risco de desvalorizar)

        Args:
            result_df: DataFrame com colunas 'predicao' (ou 'predicao_ajustada') e 'preco'

        Returns:
            DataFrame com colunas adicionais: 'mpv', 'margem_valorizacao', 'selo_valorizacao'
        """
        df = result_df.copy()

        preco = df.get('preco', pd.Series(1.0, index=df.index))

        # MPV = Mínimo de Pontos para Valorizar (aprox. preço * 0.02 por cartoleta de preço)
        # Fórmula empírica baseada na tabela oficial do Cartola FC
        df['mpv'] = preco * 2.0  # Em média, ~2 pontos por C$ de preço para valorizar

        # Usar predição ajustada se disponível, senão usar predição base
        pred_col = 'predicao_ajustada' if 'predicao_ajustada' in df.columns else 'predicao'
        predicao = df.get(pred_col, pd.Series(0.0, index=df.index))

        # Margem de valorização: diferença entre predição e MPV
        df['margem_valorizacao'] = predicao - df['mpv']

        # Classificação por faixas
        conditions = [
            predicao >= df['mpv'] * 1.5,
            predicao >= df['mpv'],
            predicao >= df['mpv'] * 0.7,
        ]
        choices = ['OTIMA ESCOLHA', 'BOA ESCOLHA', 'ARRISCADO']
        df['selo_valorizacao'] = np.select(conditions, choices, default='PODE ZICAR')

        # Score de valorização (0-100) para usar no otimizador
        max_margem = df['margem_valorizacao'].abs().max()
        if max_margem > 0:
            df['valorizacao_score'] = (
                (df['margem_valorizacao'] / max_margem).clip(-1, 1) * 50 + 50
            )
        else:
            df['valorizacao_score'] = 50.0

        logger.info(
            f"🏷️ Selos de valorização: "
            f"ÓTIMA={( df['selo_valorizacao'] == 'OTIMA ESCOLHA').sum()} | "
            f"BOA={(df['selo_valorizacao'] == 'BOA ESCOLHA').sum()} | "
            f"ARRISCADO={(df['selo_valorizacao'] == 'ARRISCADO').sum()} | "
            f"ZICAR={(df['selo_valorizacao'] == 'PODE ZICAR').sum()}"
        )

        return df

    def predict_full(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pipeline completo de predição:
        1. Predição com pesos táticos
        2. Adiciona flag de valorização
        3. Ordena por predicao_ajustada desc

        Returns:
            DataFrame completo pronto para o otimizador e para análise
        """
        result = self.predict_with_tactical_weights(df)
        result = self.add_valorizacao_flag(result)

        sort_col = 'predicao_ajustada' if 'predicao_ajustada' in result.columns else 'predicao'
        result = result.sort_values(sort_col, ascending=False).reset_index(drop=True)

        return result
