"""
Neural Network Prediction Model for Cryptocurrency Investment Recommendations
"""

import sys
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional, Tuple, Any
import logging

# Import configuration
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Import new model architectures and utilities
from backend.models.architectures import get_model_for_horizon, MODEL_REGISTRY
from backend.temporal_model_trainer import get_temporal_trainer
from backend.feature_engineering import get_feature_engineer
from backend.market_regime_detector import get_regime_detector

logger = logging.getLogger(__name__)

class EnhancedCryptoPredictor:
    """
    Enhanced Cryptocurrency Predictor with multi-horizon models and advanced features
    """
    
    def __init__(self, symbol: str, use_enhanced_features: bool = True, use_ensemble: bool = True):
        """
        Initialize the enhanced cryptocurrency predictor
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC-USD')
            use_enhanced_features: Whether to use advanced feature engineering
            use_ensemble: Whether to use ensemble models
        """
        self.symbol = symbol
        self.base_symbol = symbol.split('-')[0]
        self.model_dir = os.path.join(config.MODEL_DIRECTORY, self.base_symbol)
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Enhanced configuration
        self.use_enhanced_features = use_enhanced_features
        self.use_ensemble = use_ensemble
        
        # Initialize scalers (use RobustScaler for better outlier handling)
        self.scalers = {horizon: RobustScaler() for horizon in config.PREDICTION_HORIZONS}
        
        # Initialize models for each prediction horizon
        self.models = {horizon: None for horizon in config.PREDICTION_HORIZONS}
        
        # Initialize feature columns
        self.feature_columns = []
        
        # Initialize sequence length
        self.sequence_length = config.SEQUENCE_LENGTH
        
        # Initialize utility classes with feature selection
        self.feature_engineer = get_feature_engineer(max_features=30) if use_enhanced_features else None
        self.regime_detector = get_regime_detector() if use_enhanced_features else None
        self.temporal_trainer = get_temporal_trainer()
        
        # Model metadata
        self.model_metadata = {}
        self.confidence_intervals = {}
        
        logger.info(f"Initialized Enhanced Predictor for {symbol} (enhanced_features={use_enhanced_features}, ensemble={use_ensemble})")
        
    def load_data(self, enhanced: bool = None) -> pd.DataFrame:
        """
        Load and process cryptocurrency data with optional enhanced features
        
        Args:
            enhanced: Whether to load enhanced data (defaults to instance setting)
            
        Returns:
            DataFrame with processed data
        """
        enhanced = enhanced if enhanced is not None else self.use_enhanced_features
        
        # Try to load enhanced data first
        if enhanced:
            enhanced_file = os.path.join(config.RAW_DATA_DIRECTORY, f"{self.base_symbol}_enhanced.json")
            if os.path.exists(enhanced_file):
                try:
                    with open(enhanced_file, 'r') as f:
                        enhanced_data = json.load(f)
                    
                    # Extract primary OHLCV data
                    primary_data = None
                    if enhanced_data.get('coingecko', {}).get('ohlcv'):
                        primary_data = pd.DataFrame(enhanced_data['coingecko']['ohlcv'])
                    elif enhanced_data.get('yahoo_finance', {}).get('ohlcv'):
                        primary_data = pd.DataFrame(enhanced_data['yahoo_finance']['ohlcv'])
                    
                    if primary_data is not None and not primary_data.empty:
                        if 'timestamp' in primary_data.columns:
                            primary_data['timestamp'] = pd.to_datetime(primary_data['timestamp'])
                            primary_data.set_index('timestamp', inplace=True)
                        
                        logger.info(f"Loaded enhanced data for {self.symbol}: {len(primary_data)} records")
                        return primary_data
                except Exception as e:
                    logger.warning(f"Failed to load enhanced data: {e}")
        
        # Fallback to legacy data loading
        price_file = os.path.join(config.RAW_DATA_DIRECTORY, f"{self.base_symbol}_USD.csv")
        if os.path.exists(price_file):
            price_df = pd.read_csv(price_file)
            if 'date' in price_df.columns:
                price_df['date'] = pd.to_datetime(price_df['date'])
                price_df.set_index('date', inplace=True)
            
            logger.info(f"Loaded basic price data for {self.symbol}: {len(price_df)} records")
            return price_df
        
        raise FileNotFoundError(f"No data file found for {self.symbol}")
    
    def prepare_features(self, df: pd.DataFrame, training_date: str = None) -> pd.DataFrame:
        """
        Prepare features for the model with enhanced feature engineering
        
        Args:
            df: DataFrame with price data
            training_date: Optional training date for temporal slicing
            
        Returns:
            DataFrame with selected features
        """
        feature_df = df.copy()
        
        # Apply enhanced feature engineering if enabled
        if self.use_enhanced_features and self.feature_engineer:
            try:
                # Apply feature engineering
                enhanced_data = self.feature_engineer.apply_feature_engineering(
                    {self.symbol: feature_df}, include_cross_asset=False
                )
                feature_df = enhanced_data[self.symbol]
                
                # Apply market regime detection
                if self.regime_detector:
                    feature_df = self.regime_detector.detect_combined_regimes(feature_df)
                
                logger.info(f"Applied enhanced feature engineering: {feature_df.shape[1]} features")
                
            except Exception as e:
                logger.warning(f"Enhanced feature engineering failed: {e}")
                # Continue with basic features
        
        # Define feature columns based on available data
        exclude_cols = ['date', 'timestamp', 'symbol']
        available_cols = [col for col in feature_df.columns if col not in exclude_cols]
        
        # Use selected features if feature engineering was applied
        if self.feature_engineer and hasattr(self.feature_engineer, 'selected_features') and self.feature_engineer.selected_features:
            # Use the features selected by the feature engineer
            self.feature_columns = [col for col in self.feature_engineer.selected_features if col in available_cols]
            # Ensure we have essential OHLCV columns
            essential_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in essential_cols:
                if col in available_cols and col not in self.feature_columns:
                    self.feature_columns.append(col)
        else:
            # Use all available columns
            self.feature_columns = available_cols
        
        # Ensure we have basic OHLCV data
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in feature_df.columns]
        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
        
        logger.info(f"Selected features: {len(self.feature_columns)} columns (reduced from {len(available_cols)} available)")
        
        # Apply temporal slicing if training date provided
        if training_date and self.temporal_trainer:
            feature_df = self.temporal_trainer.slice_temporal_data(feature_df, training_date)
        
        # Drop rows with excessive NaN values
        feature_df = feature_df.dropna(thresh=len(self.feature_columns) * 0.8)
        
        # Forward fill remaining NaN values
        feature_df[self.feature_columns] = feature_df[self.feature_columns].fillna(method='ffill')
        feature_df[self.feature_columns] = feature_df[self.feature_columns].fillna(0)
        
        # Return feature columns while preserving the datetime index
        result_df = feature_df[self.feature_columns].copy()
        # Ensure the index is preserved from the original feature_df
        result_df.index = feature_df.index
        return result_df
    
    def create_sequences(self, data, target_column, prediction_horizon):
        """
        Create sequences for LSTM model
        
        Args:
            data: DataFrame with features
            target_column: Column to predict
            prediction_horizon: Number of days ahead to predict
            
        Returns:
            X: Input sequences
            y: Target values
        """
        X = []
        y = []
        
        for i in range(len(data) - self.sequence_length - prediction_horizon):
            # Input sequence
            seq = data[i:i+self.sequence_length].values
            X.append(seq)
            
            # Target value (n days ahead)
            target = data.iloc[i+self.sequence_length+prediction_horizon-1][target_column]
            y.append(target)
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        """
        Build improved LSTM model with stronger regularization
        
        Args:
            input_shape: Shape of input sequences
            
        Returns:
            Compiled Keras model
        """
        from tensorflow.keras.regularizers import l1_l2
        
        model = Sequential()
        
        # LSTM layers with stronger regularization
        model.add(LSTM(
            units=64,  # Reduced from 128
            return_sequences=True, 
            input_shape=input_shape,
            dropout=0.4,  # Increased from 0.2
            recurrent_dropout=0.3,
            kernel_regularizer=l1_l2(l1=0.001, l2=0.001)
        ))
        model.add(BatchNormalization())
        
        model.add(LSTM(
            units=32,  # Reduced from 64
            return_sequences=False,
            dropout=0.4,
            recurrent_dropout=0.3,
            kernel_regularizer=l1_l2(l1=0.001, l2=0.001)
        ))
        model.add(BatchNormalization())
        
        # Dense layers with regularization
        model.add(Dense(
            units=16,  # Reduced from 32
            activation='relu',
            kernel_regularizer=l1_l2(l1=0.001, l2=0.001)
        ))
        model.add(Dropout(0.5))  # Increased dropout
        
        # Output layer
        model.add(Dense(units=1))
        
        # Compile model with gradient clipping
        optimizer = Adam(learning_rate=config.LEARNING_RATE, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
        
        return model
    
    def train_model_temporal(self, horizon: int, training_date: str = None, 
                           progress_callback: callable = None, force_retrain: bool = False) -> Tuple[Any, Dict]:
        """
        Train model for a specific prediction horizon with temporal awareness
        
        Args:
            horizon: Number of days ahead to predict
            training_date: Date to train up to (YYYY-MM-DD format)
            progress_callback: Callback for training progress
            force_retrain: Force retraining even if cached
            
        Returns:
            Tuple of (trained_model, training_metrics)
        """
        logger.info(f"Training temporal model for {horizon}-day prediction horizon")
        
        # Use current date if training_date not specified
        if training_date is None:
            training_date = datetime.now().strftime('%Y-%m-%d')
        
        # Load and prepare data
        df = self.load_data()
        feature_df = self.prepare_features(df, training_date)
        
        # Create training configuration
        training_config = {
            'sequence_length': self.sequence_length,
            'use_enhanced_features': self.use_enhanced_features,
            'use_ensemble': self.use_ensemble,
            'feature_set_version': config.FEATURE_SET_VERSION,
            'batch_size': config.BATCH_SIZE,
            'epochs': config.EPOCHS,
            'learning_rate': config.LEARNING_RATE
        }
        
        # Get appropriate model class
        model_class = get_model_for_horizon
        
        # Use temporal trainer for point-in-time training
        try:
            model, metrics = self.temporal_trainer.train_temporal_model(
                model_class=lambda **kwargs: get_model_for_horizon(
                    horizon=horizon,
                    input_shape=(self.sequence_length, len(self.feature_columns)),
                    use_ensemble=self.use_ensemble,
                    **kwargs
                ),
                symbol=self.symbol,
                training_date=training_date,
                horizon=horizon,
                training_data=feature_df,
                training_config=training_config,
                force_retrain=force_retrain
            )
            
            # Store model and metadata
            self.models[horizon] = model
            self.model_metadata[horizon] = {
                'training_date': training_date,
                'metrics': metrics,
                'feature_count': len(self.feature_columns),
                'use_ensemble': self.use_ensemble
            }
            
            logger.info(f"Successfully trained {horizon}-day model for {self.symbol}")
            return model, metrics
            
        except Exception as e:
            logger.error(f"Failed to train {horizon}-day model: {e}")
            raise
    
    def train_model(self, horizon: int) -> Tuple[Any, Dict]:
        """
        Legacy training method for backward compatibility
        """
        return self.train_model_temporal(horizon)
    
    def load_trained_model(self, horizon):
        """
        Load a trained model for a specific prediction horizon
        
        Args:
            horizon: Number of days ahead to predict
            
        Returns:
            Loaded model
        """
        model_path = os.path.join(self.model_dir, f"model_{horizon}day.keras")
        if os.path.exists(model_path):
            model = load_model(model_path)
            self.models[horizon] = model
            print(f"Loaded model for {horizon}-day prediction horizon")
            return model
        else:
            print(f"Model file not found: {model_path}")
            return None
    
    def predict_with_confidence(self, horizon: int, latest_data: pd.DataFrame = None, 
                              prediction_date: str = None) -> Dict:
        """
        Make predictions with confidence intervals for a specific horizon
        
        Args:
            horizon: Number of days ahead to predict
            latest_data: Optional latest data to use for prediction
            prediction_date: Date to make prediction from
            
        Returns:
            Dictionary with prediction, confidence intervals, and metadata
        """
        # Load model if not already loaded
        if self.models[horizon] is None:
            try:
                self.load_trained_model(horizon)
            except:
                # Try to train model if not available
                logger.info(f"Training model for {horizon}-day prediction")
                self.train_model_temporal(horizon, training_date=prediction_date)
        
        if self.models[horizon] is None:
            logger.error(f"No trained model available for {horizon}-day prediction")
            return None
        
        # Prepare data
        if latest_data is None:
            df = self.load_data()
            feature_df = self.prepare_features(df, training_date=prediction_date)
            latest_data = feature_df.iloc[-self.sequence_length:].copy()
        
        # Ensure we have enough data
        if len(latest_data) < self.sequence_length:
            logger.warning(f"Insufficient data for prediction: {len(latest_data)} < {self.sequence_length}")
            return None
        
        # Get the model
        model = self.models[horizon]
        
        # Handle different model types
        if hasattr(model, 'scaler') and hasattr(model, 'keras_model'):
            # New TrainedModel with built-in scaler
            try:
                # Prepare data for prediction using the same transformations as training
                prediction_data = latest_data[model.feature_columns].copy()
                
                # Apply categorical mappings if they exist
                if hasattr(model, 'categorical_mappings') and model.categorical_mappings:
                    for col, mapping in model.categorical_mappings.items():
                        if col in prediction_data.columns:
                            logger.info(f"Applying categorical mapping to column '{col}' during prediction")
                            prediction_data[col] = prediction_data[col].map(mapping).fillna(-1)
                
                # Ensure all columns are numeric (same as training)
                for col in prediction_data.columns:
                    if prediction_data[col].dtype == 'object':
                        logger.warning(f"Column '{col}' still not numeric during prediction, forcing conversion")
                        prediction_data[col] = pd.to_numeric(prediction_data[col], errors='coerce').fillna(0)
                
                # Scale data using the model's scaler
                scaled_data = model.scaler.transform(prediction_data)
                X = np.array([scaled_data])
                
                # Make prediction using unscaled method
                predicted_price = model.predict_unscaled(X)[0][0]
                
                # Calculate realistic confidence intervals
                if horizon in self.model_metadata:
                    metrics = self.model_metadata[horizon]['metrics']
                    val_mae = metrics.get('val_mae', predicted_price * 0.05)
                    val_r2 = metrics.get('val_r2', 0.5)
                    val_mse = metrics.get('val_mse', (predicted_price * 0.1) ** 2)
                    
                    # Ensure R² is valid
                    val_r2 = max(-1.0, min(1.0, val_r2))
                    
                    # Calculate margin based on prediction error and horizon
                    base_margin = max(val_mae, np.sqrt(val_mse))
                    
                    # Increase margin for longer horizons (more uncertainty)
                    horizon_multiplier = 1 + (horizon - 1) * 0.1
                    margin = base_margin * horizon_multiplier
                    
                    # Apply margin as percentage of price to avoid extreme values
                    margin_pct = min(margin / predicted_price, 0.3) if predicted_price > 0 else 0.1
                    final_margin = predicted_price * margin_pct
                    
                    # Ensure bounds are positive
                    lower_bound = max(predicted_price * 0.1, predicted_price - final_margin)
                    upper_bound = predicted_price + final_margin
                    
                    # Confidence score based on R² but penalized for poor performance
                    if val_r2 > 0:
                        base_confidence = val_r2
                    else:
                        base_confidence = max(0.1, 0.5 + val_r2 * 0.1)  # Handle negative R²
                    
                    # Further reduce confidence for wide intervals
                    interval_width_penalty = min(margin_pct * 2, 0.5)
                    confidence_score = max(0.1, base_confidence * (1 - interval_width_penalty))
                else:
                    # Conservative default for models without metrics
                    margin_pct = 0.15  # 15% margin
                    margin = predicted_price * margin_pct
                    lower_bound = max(predicted_price * 0.1, predicted_price - margin)
                    upper_bound = predicted_price + margin
                    confidence_score = 0.3
                
                # Calculate prediction confidence
                confidence_width = upper_bound - lower_bound
                
                result = {
                    'predicted_price': float(predicted_price),
                    'confidence_interval': {
                        'lower': float(lower_bound),
                        'upper': float(upper_bound),
                        'width': float(confidence_width)
                    },
                    'confidence_score': float(confidence_score),
                    'horizon': horizon,
                    'prediction_date': prediction_date or datetime.now().strftime('%Y-%m-%d'),
                    'model_type': 'lstm_trained',
                    'feature_count': len(model.feature_columns),
                    'training_metrics': self.model_metadata.get(horizon, {}).get('metrics', {})
                }
                
                return result
                
            except Exception as e:
                logger.error(f"Prediction failed with TrainedModel: {e}")
                return None
                
        elif hasattr(model, 'last_price'):
            # Baseline model
            predicted_price = model.last_price
            
            # Baseline model has high uncertainty but realistic bounds
            margin_pct = 0.2  # 20% margin for baseline
            margin = predicted_price * margin_pct
            lower_bound = max(predicted_price * 0.1, predicted_price - margin)  # Ensure positive
            upper_bound = predicted_price + margin
            confidence_width = upper_bound - lower_bound
            confidence_score = 0.2  # Low confidence for baseline
            
            result = {
                'predicted_price': float(predicted_price),
                'confidence_interval': {
                    'lower': float(lower_bound),
                    'upper': float(upper_bound),
                    'width': float(confidence_width)
                },
                'confidence_score': float(confidence_score),
                'horizon': horizon,
                'prediction_date': prediction_date or datetime.now().strftime('%Y-%m-%d'),
                'model_type': 'baseline',
                'feature_count': len(self.feature_columns)
            }
            
            return result
            
        else:
            # Legacy model handling (fallback)
            try:
                # Scale data
                scaled_data = self.scalers[horizon].transform(latest_data)
                X = np.array([scaled_data])
                
                # Make prediction
                if hasattr(model, 'predict_with_confidence'):
                    # Ensemble model with confidence intervals
                    prediction = model.predict(X)[0][0]
                    lower_bound, upper_bound = model.get_confidence_intervals(X)
                    individual_preds = model.get_individual_predictions(X)
                else:
                    # Single model prediction
                    prediction = model.predict(X)[0][0]
                    # Estimate confidence based on training metrics
                    if horizon in self.model_metadata:
                        mae = self.model_metadata[horizon]['metrics'].get('final_val_loss', 0.1)
                        lower_bound = np.array([[prediction - mae]])
                        upper_bound = np.array([[prediction + mae]])
                        individual_preds = {'single_model': np.array([[prediction]])}
                    else:
                        lower_bound = upper_bound = np.array([[prediction]])
                        individual_preds = {'single_model': np.array([[prediction]])}
                
                # Inverse transform predictions
                def inverse_transform_price(scaled_value):
                    if 'close' in self.feature_columns:
                        dummy = np.zeros((1, len(self.feature_columns)))
                        close_idx = self.feature_columns.index('close')
                        dummy[0, close_idx] = scaled_value
                        dummy_inverse = self.scalers[horizon].inverse_transform(dummy)
                        return dummy_inverse[0, close_idx]
                    else:
                        # Fallback: assume close is the last feature
                        dummy = np.zeros((1, len(self.feature_columns)))
                        dummy[0, -1] = scaled_value
                        dummy_inverse = self.scalers[horizon].inverse_transform(dummy)
                        return dummy_inverse[0, -1]
                
                # Transform all predictions to actual prices
                predicted_price = inverse_transform_price(prediction)
                lower_price = inverse_transform_price(lower_bound[0][0])
                upper_price = inverse_transform_price(upper_bound[0][0])
                
                # Calculate prediction confidence
                confidence_width = upper_price - lower_price
                confidence_score = max(0, min(1, 1 - (confidence_width / predicted_price))) if predicted_price > 0 else 0.5
                
                result = {
                    'predicted_price': float(predicted_price),
                    'confidence_interval': {
                        'lower': float(lower_price),
                        'upper': float(upper_price),
                        'width': float(confidence_width)
                    },
                    'confidence_score': float(confidence_score),
                    'horizon': horizon,
                    'prediction_date': prediction_date or datetime.now().strftime('%Y-%m-%d'),
                    'model_type': 'ensemble' if self.use_ensemble else 'single',
                    'feature_count': len(self.feature_columns)
                }
                
                if self.use_ensemble and hasattr(model, 'get_individual_predictions'):
                    individual_prices = {}
                    for name, pred in individual_preds.items():
                        individual_prices[name] = float(inverse_transform_price(pred[0][0]))
                    result['individual_predictions'] = individual_prices
                
                return result
                
            except Exception as e:
                logger.error(f"Legacy model prediction failed: {e}")
                return None
    
    def predict(self, horizon: int, latest_data: pd.DataFrame = None) -> Optional[float]:
        """
        Legacy prediction method for backward compatibility
        """
        result = self.predict_with_confidence(horizon, latest_data)
        return result['predicted_price'] if result else None
    
    def predict_multiple_horizons(self, horizons: List[int] = None, current_date: str = None) -> Dict:
        """
        Make predictions for multiple horizons with confidence intervals
        
        Args:
            horizons: List of prediction horizons (days ahead)
            current_date: Date to start predictions from (defaults to latest available)
            
        Returns:
            Dictionary with predictions and confidence intervals for each horizon
        """
        horizons = horizons or config.PREDICTION_HORIZONS
        predictions = {}
        
        # Load data
        df = self.load_data()
        feature_df = self.prepare_features(df, training_date=current_date)
        
        # Determine base date
        if current_date is None:
            latest_data = feature_df.iloc[-self.sequence_length:].copy()
            base_date = feature_df.index[-1] if hasattr(feature_df.index[-1], 'strftime') else datetime.now()
        else:
            current_datetime = pd.to_datetime(current_date)
            try:
                # Slice data up to current date
                sliced_df = feature_df[feature_df.index <= current_datetime]
                latest_data = sliced_df.iloc[-self.sequence_length:].copy()
                base_date = current_datetime
            except Exception as e:
                logger.warning(f"Error slicing data for date {current_date}: {e}")
                latest_data = feature_df.iloc[-self.sequence_length:].copy()
                base_date = feature_df.index[-1] if hasattr(feature_df.index[-1], 'strftime') else datetime.now()
        
        # Make predictions for each horizon
        for horizon in horizons:
            try:
                prediction_result = self.predict_with_confidence(horizon, latest_data, current_date)
                if prediction_result is not None:
                    prediction_date = base_date + pd.Timedelta(days=horizon)
                    predictions[horizon] = {
                        **prediction_result,
                        'prediction_date': prediction_date.strftime('%Y-%m-%d') if hasattr(prediction_date, 'strftime') else str(prediction_date),
                        'base_date': base_date.strftime('%Y-%m-%d') if hasattr(base_date, 'strftime') else str(base_date)
                    }
                    
                    logger.debug(f"Prediction for {horizon}d: {prediction_result['predicted_price']:.2f} ± {prediction_result['confidence_interval']['width']:.2f}")
            except Exception as e:
                logger.error(f"Failed to predict for {horizon}-day horizon: {e}")
                predictions[horizon] = {
                    'error': str(e),
                    'predicted_price': None,
                    'confidence_interval': None
                }
        
        return predictions
    
    def generate_prediction_series(self, current_date, days_ahead=30):
        """
        Generate a series of predictions for charting
        
        Args:
            current_date: Starting date for predictions
            days_ahead: Number of days to predict ahead
            
        Returns:
            DataFrame with prediction dates and values
        """
        predictions = []
        
        # Load data and get historical context
        df = self.load_data()
        feature_df = self.prepare_features(df)
        
        if isinstance(current_date, str):
            current_date = pd.to_datetime(current_date)
        
        try:
            # Get data up to current date
            historical_data = feature_df[feature_df.index <= current_date]
            latest_sequence = historical_data.iloc[-self.sequence_length:].copy()
            
            # Generate predictions for 7, 14, and 30 day horizons
            horizons = [7, 14, 30]
            for horizon in horizons:
                if horizon <= days_ahead:
                    predicted_price = self.predict(horizon, latest_sequence)
                    if predicted_price is not None:
                        prediction_date = current_date + pd.Timedelta(days=horizon)
                        predictions.append({
                            'date': prediction_date,
                            'predicted_price': predicted_price,
                            'horizon': horizon,
                            'type': 'prediction'
                        })
            
            return pd.DataFrame(predictions)
            
        except Exception as e:
            print(f"Error generating prediction series: {str(e)}")
            return pd.DataFrame()
    
    def generate_recommendations(self):
        """
        Generate investment recommendations based on predictions
        
        Returns:
            Dictionary with recommendations
        """
        recommendations = {}
        
        # Load latest data
        df = self.load_data()
        feature_df = self.prepare_features(df)
        latest_data = feature_df.iloc[-self.sequence_length:].copy()
        
        # Current price
        current_price = feature_df.iloc[-1]['close']
        
        for horizon in config.PREDICTION_HORIZONS:
            # Make prediction
            predicted_price = self.predict(horizon, latest_data)
            
            if predicted_price is not None:
                # Calculate expected return
                expected_return = (predicted_price - current_price) / current_price * 100
                
                # Determine recommendation
                if expected_return > 5:
                    action = "BUY"
                    strength = "STRONG"
                elif expected_return > 1:
                    action = "BUY"
                    strength = "MODERATE"
                elif expected_return > -1:
                    action = "HOLD"
                    strength = "NEUTRAL"
                elif expected_return > -5:
                    action = "SELL"
                    strength = "MODERATE"
                else:
                    action = "SELL"
                    strength = "STRONG"
                
                # Add to recommendations
                recommendations[horizon] = {
                    'current_price': float(current_price),
                    'predicted_price': float(predicted_price),
                    'expected_return': float(expected_return),
                    'action': action,
                    'strength': strength,
                    'prediction_date': (datetime.now() + timedelta(days=horizon)).strftime('%Y-%m-%d')
                }
        
        return recommendations
    
    def generate_enhanced_recommendations(self, predictions: Dict = None) -> Dict:
        """
        Generate enhanced investment recommendations with confidence analysis
        
        Args:
            predictions: Optional pre-computed predictions
            
        Returns:
            Dictionary with enhanced recommendations
        """
        if predictions is None:
            predictions = self.predict_multiple_horizons()
        
        recommendations = {}
        
        # Get current price from latest data
        try:
            df = self.load_data()
            current_price = df['close'].iloc[-1] if 'close' in df.columns else df.iloc[-1, -1]
        except:
            current_price = None
        
        for horizon, pred_data in predictions.items():
            if 'error' in pred_data or pred_data.get('predicted_price') is None:
                continue
            
            predicted_price = pred_data['predicted_price']
            confidence_interval = pred_data.get('confidence_interval', {})
            confidence_score = pred_data.get('confidence_score', 0.5)
            
            if current_price is None:
                continue
            
            # Calculate expected return
            expected_return = (predicted_price - current_price) / current_price * 100
            
            # Calculate confidence-adjusted return
            confidence_adjusted_return = expected_return * confidence_score
            
            # Risk assessment based on confidence interval width
            ci_width = confidence_interval.get('width', 0)
            risk_ratio = ci_width / predicted_price if predicted_price > 0 else 1
            
            # Determine action based on confidence-adjusted return and risk
            if confidence_adjusted_return > 5 and confidence_score > 0.7:
                action = "STRONG_BUY"
                strength = "HIGH"
            elif confidence_adjusted_return > 2 and confidence_score > 0.6:
                action = "BUY"
                strength = "MODERATE"
            elif confidence_adjusted_return > 0 and confidence_score > 0.5:
                action = "WEAK_BUY"
                strength = "LOW"
            elif confidence_adjusted_return > -2 and confidence_score > 0.5:
                action = "HOLD"
                strength = "NEUTRAL"
            elif confidence_adjusted_return > -5 and confidence_score > 0.6:
                action = "WEAK_SELL"
                strength = "LOW"
            elif confidence_adjusted_return > -10 and confidence_score > 0.7:
                action = "SELL"
                strength = "MODERATE"
            else:
                action = "STRONG_SELL"
                strength = "HIGH"
            
            # Risk level assessment
            if risk_ratio < 0.05:
                risk_level = "LOW"
            elif risk_ratio < 0.15:
                risk_level = "MODERATE"
            else:
                risk_level = "HIGH"
            
            recommendations[horizon] = {
                'current_price': float(current_price),
                'predicted_price': float(predicted_price),
                'expected_return': float(expected_return),
                'confidence_adjusted_return': float(confidence_adjusted_return),
                'confidence_score': float(confidence_score),
                'confidence_interval': confidence_interval,
                'action': action,
                'strength': strength,
                'risk_level': risk_level,
                'risk_ratio': float(risk_ratio),
                'prediction_date': pred_data.get('prediction_date'),
                'base_date': pred_data.get('base_date'),
                'model_type': pred_data.get('model_type', 'unknown'),
                'feature_count': pred_data.get('feature_count', 0)
            }
            
            # Add individual model predictions if available
            if 'individual_predictions' in pred_data:
                recommendations[horizon]['individual_predictions'] = pred_data['individual_predictions']
        
        return recommendations
    
    def save_recommendations(self, recommendations):
        """
        Save recommendations to file
        
        Args:
            recommendations: Dictionary with recommendations
        """
        # Create recommendations directory
        recommendations_dir = os.path.join(config.DATA_DIRECTORY, 'recommendations')
        os.makedirs(recommendations_dir, exist_ok=True)
        
        # Save recommendations
        filename = os.path.join(recommendations_dir, f"{self.base_symbol}_recommendations.json")
        with open(filename, 'w') as f:
            json.dump(recommendations, f, indent=4)
        
        print(f"Saved recommendations to {filename}")
        
        return filename

def train_all_models_temporal(training_date: str = None, use_enhanced: bool = True, 
                             use_ensemble: bool = True, progress_callback: callable = None) -> Dict:
    """
    Train enhanced models for all cryptocurrencies and prediction horizons
    
    Args:
        training_date: Date to train up to (defaults to current date)
        use_enhanced: Whether to use enhanced features
        use_ensemble: Whether to use ensemble models
        progress_callback: Callback for overall progress
        
    Returns:
        Dictionary with training results
    """
    training_date = training_date or datetime.now().strftime('%Y-%m-%d')
    results = {}
    
    total_tasks = len(config.CRYPTO_SYMBOLS) * len(config.PREDICTION_HORIZONS)
    completed_tasks = 0
    
    for symbol in config.CRYPTO_SYMBOLS:
        try:
            predictor = EnhancedCryptoPredictor(
                symbol, 
                use_enhanced_features=use_enhanced,
                use_ensemble=use_ensemble
            )
            
            symbol_results = {}
            
            for horizon in config.PREDICTION_HORIZONS:
                try:
                    logger.info(f"Training {symbol} for {horizon}-day horizon")
                    
                    # Individual model progress callback
                    def model_progress_callback(epoch, logs):
                        if progress_callback:
                            overall_progress = (completed_tasks + epoch / config.EPOCHS) / total_tasks
                            progress_callback({
                                'overall_progress': overall_progress,
                                'current_symbol': symbol,
                                'current_horizon': horizon,
                                'epoch': epoch,
                                'logs': logs
                            })
                    
                    model, metrics = predictor.train_model_temporal(
                        horizon, training_date, model_progress_callback
                    )
                    
                    symbol_results[horizon] = {
                        'success': True,
                        'metrics': metrics
                    }
                    
                except Exception as e:
                    logger.error(f"Failed to train {symbol} {horizon}d model: {e}")
                    symbol_results[horizon] = {
                        'success': False,
                        'error': str(e)
                    }
                
                completed_tasks += 1
                
                if progress_callback:
                    progress_callback({
                        'overall_progress': completed_tasks / total_tasks,
                        'completed_tasks': completed_tasks,
                        'total_tasks': total_tasks
                    })
            
            results[symbol] = symbol_results
            
        except Exception as e:
            logger.error(f"Failed to initialize predictor for {symbol}: {e}")
            results[symbol] = {'error': str(e)}
    
    return results

def generate_all_recommendations_enhanced(current_date: str = None, use_enhanced: bool = True) -> Dict:
    """
    Generate enhanced recommendations for all cryptocurrencies
    
    Args:
        current_date: Date to generate recommendations from
        use_enhanced: Whether to use enhanced features
        
    Returns:
        Dictionary with enhanced recommendations for all cryptocurrencies
    """
    all_recommendations = {}
    
    for symbol in config.CRYPTO_SYMBOLS:
        try:
            predictor = EnhancedCryptoPredictor(
                symbol, 
                use_enhanced_features=use_enhanced,
                use_ensemble=True
            )
            
            # Generate predictions with confidence intervals
            predictions = predictor.predict_multiple_horizons(current_date=current_date)
            
            # Generate enhanced recommendations
            recommendations = predictor.generate_enhanced_recommendations(predictions)
            
            # Save recommendations
            predictor.save_recommendations(recommendations)
            
            all_recommendations[symbol] = recommendations
            
        except Exception as e:
            logger.error(f"Failed to generate recommendations for {symbol}: {e}")
            all_recommendations[symbol] = {'error': str(e)}
    
    return all_recommendations

# Maintain backward compatibility
def train_all_models():
    """Legacy training function for backward compatibility"""
    return train_all_models_temporal(use_enhanced=False, use_ensemble=False)

def generate_all_recommendations():
    """Legacy recommendation function for backward compatibility"""
    return generate_all_recommendations_enhanced(use_enhanced=False)

# Alias for the legacy class
CryptoPredictor = EnhancedCryptoPredictor

if __name__ == "__main__":
    # Create model directory
    os.makedirs(config.MODEL_DIRECTORY, exist_ok=True)
    
    # Train all models
    train_all_models()
    
    # Generate recommenda<response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>