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
    Modelo de predição de pontuação usando ensemble
    """
    
    def __init__(self, model_type: str = 'rf'):
        self.model_type = model_type
        self.model = None
        self.feature_columns = None
        
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
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepara features para treinamento/predição
        """
        # Features a serem usadas
        feature_cols = [
            # Médias móveis
            'pontos_media_3', 'pontos_media_5', 'pontos_media_10',
            'pontos_std_3', 'pontos_std_5',
            'pontos_max_5', 'pontos_min_5',
            
            # Forma
            'tendencia', 'sequencia_positiva', 'pontos_por_90min',
            
            # Adversário
            'forca_adversario', 'mando_casa',
            
            # Scouts
            'gols_por_finalizacao', 'taxa_assistencia', 'eficiencia_defensiva',
            
            # Preço
            'custo_beneficio', 'distancia_mpv', 'variacao_acumulada_5',
            
            # Básicas
            'posicao_id', 'minutos_jogados', 'jogos', 'media', 'preco'
        ]
        
        # Filtrar colunas que existem
        available_cols = [col for col in feature_cols if col in df.columns]
        
        X = df[available_cols].copy()
        y = df['pontos'].copy() if 'pontos' in df.columns else None
        
        self.feature_columns = available_cols
        
        return X, y
    
    def train(self, df: pd.DataFrame, validate: bool = True) -> Dict:
        """
        Treina modelo com validação temporal
        """
        X, y = self.prepare_features(df)
        
        if y is None:
            raise ValueError("Target 'pontos' not found in dataframe!")

        logger.info(f"Treinando modelo com {len(X)} amostras e {len(X.columns)} features")
        
        if validate and len(X) > 10:
            # Time Series Cross-Validation
            tscv = TimeSeriesSplit(n_splits=min(5, len(X)//2))
            try:
                scores = cross_val_score(
                    self.model, X, y,
                    cv=tscv,
                    scoring='neg_mean_absolute_error',
                    n_jobs=-1
                )
                logger.info(f"MAE Cross-Validation: {-scores.mean():.2f} (+/- {scores.std():.2f})")
            except Exception as e:
                logger.warning(f"Erro no Cross-Validation: {e}")
        
        # Treinar no dataset completo
        self.model.fit(X, y)
        
        # Métricas no conjunto de treino
        y_pred = self.model.predict(X)
        
        metrics = {
            'mae': mean_absolute_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'r2': r2_score(y, y_pred)
        }
        
        logger.info(f"Métricas de treino - MAE: {metrics['mae']:.2f}, RMSE: {metrics['rmse']:.2f}, R²: {metrics['r2']:.3f}")
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            logger.info(f"\nTop 10 features mais importantes:\n{feature_importance.head(10)}")
        
        return metrics
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prediz pontuação para próxima rodada
        """
        if self.model is None:
            raise ValueError("Modelo não foi treinado ainda!")
        
        X, _ = self.prepare_features(df)
        
        # Garantir que usa as mesmas features do treino
        X = X[self.feature_columns]
        
        predictions = self.model.predict(X)
        
        # Não permitir predições negativas
        predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def predict_with_confidence(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prediz com intervalo de confiança (para Random Forest)
        """
        predictions = self.predict(df)
        
        result_df = df[['atleta_id', 'apelido', 'posicao_id', 'clube_id', 'preco']].copy()
        result_df['predicao'] = predictions
        
        # Se for Random Forest, calcular intervalo de confiança
        if self.model_type == 'rf' and hasattr(self.model, 'estimators_'):
            X, _ = self.prepare_features(df)
            X = X[self.feature_columns]
            
            # Predições de cada árvore
            all_predictions = np.array([tree.predict(X) for tree in self.model.estimators_])
            
            result_df['predicao_std'] = all_predictions.std(axis=0)
            result_df['predicao_min'] = all_predictions.min(axis=0)
            result_df['predicao_max'] = all_predictions.max(axis=0)
        
        return result_df
    
    def save_model(self, filepath: str):
        """Salva modelo treinado"""
        joblib.dump({
            'model': self.model,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type
        }, filepath)
        logger.info(f"Modelo salvo em: {filepath}")
    
    def load_model(self, filepath: str):
        """Carrega modelo treinado"""
        data = joblib.load(filepath)
        self.model = data['model']
        self.feature_columns = data['feature_columns']
        self.model_type = data['model_type']
        logger.info(f"Modelo carregado de: {filepath}")
