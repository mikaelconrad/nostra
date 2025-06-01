"""
Temporal-Aware Model Training System
Implements point-in-time training, efficient data slicing for any selected date,
training cache system, and progress callbacks for UI integration
"""

import os
import pickle
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable, Any
import logging
import json
from pathlib import Path
import threading
import time

# TensorFlow and ML imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import config

logger = logging.getLogger(__name__)

class TemporalModelTrainer:
    """Point-in-time model training with caching and progress tracking"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or config.MODEL_CACHE_DIRECTORY
        self.training_cache = {}
        self.progress_callbacks = []
        self.current_training_session = None
        self.lock = threading.Lock()
        
        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Load existing cache index
        self.cache_index_file = os.path.join(self.cache_dir, 'cache_index.json')
        self.cache_index = self._load_cache_index()
        
    def _load_cache_index(self) -> Dict:
        """Load cache index from disk"""
        if os.path.exists(self.cache_index_file):
            try:
                with open(self.cache_index_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache index: {e}")
        return {}
    
    def _save_cache_index(self):
        """Save cache index to disk"""
        try:
            with open(self.cache_index_file, 'w') as f:
                json.dump(self.cache_index, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save cache index: {e}")
    
    def add_progress_callback(self, callback: Callable[[Dict], None]):
        """Add progress callback for UI updates"""
        self.progress_callbacks.append(callback)
    
    def _notify_progress(self, progress_data: Dict):
        """Notify all progress callbacks"""
        for callback in self.progress_callbacks:
            try:
                callback(progress_data)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")
    
    def _generate_cache_key(self, symbol: str, training_date: str, horizon: int, 
                          config_hash: str) -> str:
        """Generate unique cache key for model"""
        key_string = f"{symbol}_{training_date}_{horizon}_{config_hash}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _generate_config_hash(self, training_config: Dict) -> str:
        """Generate hash of training configuration"""
        # Create deterministic hash of training parameters
        config_str = json.dumps(training_config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def slice_temporal_data(self, df: pd.DataFrame, training_date: str, 
                          lookback_days: int = None, 
                          ensure_no_future_leak: bool = True) -> pd.DataFrame:
        """
        Slice data for point-in-time training ensuring no future data leakage
        
        Args:
            df: DataFrame with time series data
            training_date: Date to train up to (YYYY-MM-DD format)
            lookback_days: Number of days to look back (None = all available)
            ensure_no_future_leak: Ensure no data after training_date
            
        Returns:
            Sliced DataFrame for training
        """
        logger.debug(f"Slicing temporal data for {training_date}")
        
        # Convert training_date to datetime
        training_datetime = pd.to_datetime(training_date)
        
        # Work with a copy
        df_work = df.copy()
        
        # Determine date column and handle different date formats
        was_datetime_index = isinstance(df.index, pd.DatetimeIndex)
        
        if was_datetime_index:
            # Data already has datetime index - slice directly
            if ensure_no_future_leak:
                df_work = df_work[df_work.index <= training_datetime]
            
            # Apply lookback window if specified
            if lookback_days is not None:
                start_date = training_datetime - timedelta(days=lookback_days)
                df_work = df_work[df_work.index >= start_date]
                
        elif 'date' in df.columns:
            df_work['date'] = pd.to_datetime(df_work['date'])
            if ensure_no_future_leak:
                df_work = df_work[df_work['date'] <= training_datetime]
            
            # Apply lookback window if specified
            if lookback_days is not None:
                start_date = training_datetime - timedelta(days=lookback_days)
                df_work = df_work[df_work['date'] >= start_date]
                
            df_work = df_work.sort_values('date').set_index('date')
            
        elif 'timestamp' in df.columns:
            df_work['timestamp'] = pd.to_datetime(df_work['timestamp'])
            if ensure_no_future_leak:
                df_work = df_work[df_work['timestamp'] <= training_datetime]
            
            # Apply lookback window if specified
            if lookback_days is not None:
                start_date = training_datetime - timedelta(days=lookback_days)
                df_work = df_work[df_work['timestamp'] >= start_date]
                
            df_work = df_work.sort_values('timestamp').set_index('timestamp')
        
        else:
            logger.warning("No recognizable date column found, using data as-is")
            if ensure_no_future_leak and was_datetime_index:
                df_work = df_work[df_work.index <= training_datetime]
        
        if isinstance(df_work.index, pd.DatetimeIndex):
            logger.debug(f"Sliced data: {len(df_work)} records from {df_work.index.min()} to {df_work.index.max()}")
        else:
            logger.debug(f"Sliced data: {len(df_work)} records")
        return df_work
    
    def create_sequences(self, data: np.ndarray, target_data: np.ndarray, 
                        sequence_length: int, prediction_horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training
        
        Args:
            data: Feature data array
            target_data: Target variable data (typically close prices)
            sequence_length: Length of input sequences
            prediction_horizon: Number of days ahead to predict
            
        Returns:
            X: Input sequences, y: Target values
        """
        X, y = [], []
        
        for i in range(len(data) - sequence_length - prediction_horizon + 1):
            # Input sequence
            seq = data[i:i + sequence_length]
            X.append(seq)
            
            # Target value (prediction_horizon days ahead)
            target_idx = i + sequence_length + prediction_horizon - 1
            if target_idx < len(target_data):
                y.append(target_data[target_idx])
        
        return np.array(X), np.array(y)
    
    def build_lstm_model(self, input_shape: Tuple[int, int], 
                        training_config: Dict) -> Sequential:
        """
        Build LSTM model for cryptocurrency prediction
        
        Args:
            input_shape: Shape of input sequences (sequence_length, n_features)
            training_config: Training configuration parameters
            
        Returns:
            Compiled Keras model
        """
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(
            units=128, 
            return_sequences=True, 
            input_shape=input_shape,
            dropout=0.2,
            recurrent_dropout=0.2
        ))
        model.add(BatchNormalization())
        
        # Second LSTM layer
        model.add(LSTM(
            units=64, 
            return_sequences=False,
            dropout=0.2,
            recurrent_dropout=0.2
        ))
        model.add(BatchNormalization())
        
        # Dense layers
        model.add(Dense(units=32, activation='relu'))
        model.add(Dropout(0.3))
        
        model.add(Dense(units=16, activation='relu'))
        model.add(Dropout(0.2))
        
        # Output layer
        model.add(Dense(units=1))
        
        # Compile model
        optimizer = Adam(learning_rate=training_config.get('learning_rate', 0.001))
        model.compile(
            optimizer=optimizer, 
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        return model

    def train_temporal_model(self, model_class, symbol: str, training_date: str,
                           horizon: int, training_data: pd.DataFrame, 
                           training_config: Dict, force_retrain: bool = False) -> Tuple[Any, Dict]:
        """
        Train a temporal model with point-in-time data slicing
        
        Args:
            model_class: Model class constructor
            symbol: Trading symbol
            training_date: Point-in-time training date (string format)
            horizon: Prediction horizon in days
            training_data: Full dataset with features
            training_config: Training configuration
            force_retrain: Force retraining even if cached
            
        Returns:
            Tuple of (trained_model, metrics)
        """
        logger.info(f"Training temporal model for {symbol} {horizon}d horizon at {training_date}")
        
        try:
            # Convert training_date to datetime if it's a string
            if isinstance(training_date, str):
                training_datetime = pd.to_datetime(training_date)
            else:
                training_datetime = training_date
            
            # Slice data up to training date (preventing future data leakage)
            sliced_data = self.slice_temporal_data(training_data, training_date)
            
            # Ensure we have enough data for training (reduced requirements)
            sequence_length = training_config.get('sequence_length', config.SEQUENCE_LENGTH)
            min_samples_needed = sequence_length + horizon + 20  # Reduced from 100 to 20
            
            if len(sliced_data) < min_samples_needed:
                logger.warning(f"Insufficient data for training: {len(sliced_data)} < {min_samples_needed}")
                # Return a simple baseline model that predicts the last known price
                return self._create_baseline_model(symbol, horizon, sliced_data)
            
            # Prepare features and target
            feature_columns = [col for col in sliced_data.columns if col not in ['date', 'timestamp']]
            
            # Ensure we have a 'close' column for target
            if 'close' not in feature_columns:
                logger.error("No 'close' column found in training data")
                return self._create_baseline_model(symbol, horizon, sliced_data)
            
            # Handle categorical features by converting to numeric
            df_numeric = sliced_data[feature_columns].copy()
            
            # Convert categorical columns to numeric and save mappings
            categorical_columns = []
            categorical_mappings = {}
            for col in df_numeric.columns:
                if df_numeric[col].dtype == 'object':
                    categorical_columns.append(col)
                    logger.info(f"Converting categorical column '{col}' to numeric")
                    
                    # Use label encoding for categorical features
                    unique_values = df_numeric[col].unique()
                    value_map = {val: idx for idx, val in enumerate(unique_values) if pd.notna(val)}
                    categorical_mappings[col] = value_map
                    df_numeric[col] = df_numeric[col].map(value_map).fillna(-1)
            
            if categorical_columns:
                logger.info(f"Converted {len(categorical_columns)} categorical columns: {categorical_columns}")
            
            # Ensure all columns are numeric
            for col in df_numeric.columns:
                if df_numeric[col].dtype == 'object':
                    logger.warning(f"Column '{col}' still not numeric, forcing conversion")
                    df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce').fillna(0)
            
            # Scale the data
            scaler = RobustScaler()
            try:
                scaled_data = scaler.fit_transform(df_numeric)
            except Exception as e:
                logger.error(f"Scaling failed: {e}")
                # Try with only numeric columns
                numeric_cols = df_numeric.select_dtypes(include=[np.number]).columns
                logger.info(f"Using only numeric columns: {list(numeric_cols)}")
                scaled_data = scaler.fit_transform(df_numeric[numeric_cols])
                feature_columns = list(numeric_cols)
            
            # Get target data (close prices)
            close_idx = feature_columns.index('close')
            target_data = scaled_data[:, close_idx]
            
            # Create sequences for LSTM training
            X, y = self.create_sequences(
                scaled_data, 
                target_data, 
                sequence_length, 
                horizon
            )
            
            if len(X) == 0:
                logger.warning("No training sequences created")
                return self._create_baseline_model(symbol, horizon, sliced_data)
            
            logger.info(f"Created {len(X)} training sequences with shape {X.shape}")
            
            # Improved temporal split with minimum validation size
            min_val_size = max(50, int(len(X) * 0.15))  # At least 50 samples or 15%
            max_train_size = len(X) - min_val_size
            train_size = min(int(len(X) * 0.8), max_train_size)
            
            X_train, X_val = X[:train_size], X[train_size:]
            y_train, y_val = y[:train_size], y[train_size:]
            
            logger.info(f"Data split: Train={len(X_train)}, Val={len(X_val)} (horizon={horizon}d)")
            
            # Check for data quality issues
            if len(X_val) < 20:
                logger.warning(f"Very small validation set ({len(X_val)} samples) - results may be unreliable")
            
            # Build model
            input_shape = (sequence_length, len(feature_columns))
            model = self.build_lstm_model(input_shape, training_config)
            
            # Setup callbacks
            model_dir = f"models/{symbol.replace('-', '')}"
            os.makedirs(model_dir, exist_ok=True)
            model_filepath = f"{model_dir}/model_{horizon}day.keras"
            
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True,
                    verbose=1
                ),
                ModelCheckpoint(
                    model_filepath,
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=1
                )
            ]
            
            # Train the model with reduced epochs and better regularization
            epochs = min(training_config.get('epochs', config.EPOCHS), 15)  # Cap at 15 epochs
            batch_size = min(training_config.get('batch_size', config.BATCH_SIZE), 32)  # Smaller batches
            
            logger.info(f"Training model with {len(X_train)} samples for {epochs} epochs, batch_size={batch_size}")
            
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
            
            # Calculate final metrics
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            
            # Inverse transform predictions for meaningful metrics
            # Create dummy array for inverse transform
            dummy_train = np.zeros((len(train_pred), len(feature_columns)))
            dummy_train[:, close_idx] = train_pred.flatten()
            train_pred_unscaled = scaler.inverse_transform(dummy_train)[:, close_idx]
            
            dummy_val = np.zeros((len(val_pred), len(feature_columns)))
            dummy_val[:, close_idx] = val_pred.flatten()
            val_pred_unscaled = scaler.inverse_transform(dummy_val)[:, close_idx]
            
            # Get actual unscaled values
            dummy_train_actual = np.zeros((len(y_train), len(feature_columns)))
            dummy_train_actual[:, close_idx] = y_train
            y_train_unscaled = scaler.inverse_transform(dummy_train_actual)[:, close_idx]
            
            dummy_val_actual = np.zeros((len(y_val), len(feature_columns)))
            dummy_val_actual[:, close_idx] = y_val
            y_val_unscaled = scaler.inverse_transform(dummy_val_actual)[:, close_idx]
            
            # Calculate metrics
            train_mse = mean_squared_error(y_train_unscaled, train_pred_unscaled)
            train_mae = mean_absolute_error(y_train_unscaled, train_pred_unscaled)
            train_r2 = r2_score(y_train_unscaled, train_pred_unscaled)
            
            val_mse = mean_squared_error(y_val_unscaled, val_pred_unscaled)
            val_mae = mean_absolute_error(y_val_unscaled, val_pred_unscaled)
            val_r2 = r2_score(y_val_unscaled, val_pred_unscaled)
            
            # Check for severe overfitting
            overfitting_ratio = val_mse / train_mse if train_mse > 0 else float('inf')
            if overfitting_ratio > 3.0:
                logger.warning(f"Severe overfitting detected: val_mse/train_mse = {overfitting_ratio:.2f}")
            elif overfitting_ratio > 2.0:
                logger.warning(f"Moderate overfitting detected: val_mse/train_mse = {overfitting_ratio:.2f}")
            
            metrics = {
                'train_mse': float(train_mse),
                'train_mae': float(train_mae),
                'train_r2': float(train_r2),
                'val_mse': float(val_mse),
                'val_mae': float(val_mae),
                'val_r2': float(val_r2),
                'overfitting_ratio': float(overfitting_ratio),
                'final_train_loss': float(history.history['loss'][-1]),
                'final_val_loss': float(history.history['val_loss'][-1]),
                'training_samples': len(X_train),
                'validation_samples': len(X_val),
                'epochs_trained': len(history.history['loss']),
                'sequence_length': sequence_length,
                'horizon': horizon,
                'feature_count': len(feature_columns),
                'data_quality_score': 1.0 / max(1.0, overfitting_ratio - 1.0)  # Quality decreases with overfitting
            }
            
            # Create a wrapper class that includes the scaler and categorical mappings
            class TrainedModel:
                def __init__(self, keras_model, scaler, feature_columns, close_idx, categorical_mappings):
                    self.keras_model = keras_model
                    self.scaler = scaler
                    self.feature_columns = feature_columns
                    self.close_idx = close_idx
                    self.categorical_mappings = categorical_mappings
                    
                def predict(self, X):
                    """Make predictions using the trained model"""
                    return self.keras_model.predict(X)
                    
                def predict_unscaled(self, X_scaled):
                    """Make predictions and inverse transform to actual prices"""
                    pred_scaled = self.keras_model.predict(X_scaled)
                    
                    # Inverse transform
                    dummy = np.zeros((len(pred_scaled), len(self.feature_columns)))
                    dummy[:, self.close_idx] = pred_scaled.flatten()
                    pred_unscaled = self.scaler.inverse_transform(dummy)[:, self.close_idx]
                    
                    return pred_unscaled.reshape(-1, 1)
                
                def save(self, filepath):
                    """Save the model and scaler"""
                    self.keras_model.save(filepath)
                    
                    # Save scaler separately
                    scaler_path = filepath.replace('.keras', '_scaler.pkl')
                    with open(scaler_path, 'wb') as f:
                        pickle.dump({
                            'scaler': self.scaler,
                            'feature_columns': self.feature_columns,
                            'close_idx': self.close_idx,
                            'categorical_mappings': self.categorical_mappings
                        }, f)
                    
                    logger.info(f"Saved model to {filepath} and scaler to {scaler_path}")
            
            # Create the wrapped model
            trained_model = TrainedModel(model, scaler, feature_columns, close_idx, categorical_mappings)
            
            logger.info(f"Completed training for {symbol} {horizon}d model - Val MSE: {val_mse:.4f}, Val R2: {val_r2:.4f}")
            return trained_model, metrics
            
        except Exception as e:
            logger.error(f"Training failed for {symbol} {horizon}d: {e}")
            # Return baseline model as fallback
            return self._create_baseline_model(symbol, horizon, training_data)
    
    def _create_baseline_model(self, symbol: str, horizon: int, data: pd.DataFrame) -> Tuple[Any, Dict]:
        """Create a simple baseline model that predicts the last known price"""
        
        class BaselineModel:
            def __init__(self, last_price):
                self.last_price = last_price
                
            def predict(self, X):
                # Return last known price for all predictions
                return np.full((len(X), 1), self.last_price)
                
            def predict_unscaled(self, X):
                return self.predict(X)
                
            def save(self, filepath):
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                with open(filepath.replace('.keras', '_baseline.json'), 'w') as f:
                    json.dump({'last_price': self.last_price, 'type': 'baseline'}, f)
        
        # Get last known price
        last_price = data['close'].iloc[-1] if 'close' in data.columns else 0
        
        model = BaselineModel(last_price)
        
        metrics = {
            'type': 'baseline',
            'last_price': float(last_price),
            'training_samples': len(data),
            'baseline_prediction': float(last_price)
        }
        
        logger.info(f"Created baseline model for {symbol} {horizon}d predicting ${last_price:.2f}")
        return model, metrics


# Singleton instance
_temporal_trainer = None

def get_temporal_trainer() -> TemporalModelTrainer:
    """Get singleton temporal model trainer instance"""
    global _temporal_trainer
    if _temporal_trainer is None:
        _temporal_trainer = TemporalModelTrainer()
    return _temporal_trainer