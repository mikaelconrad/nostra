"""
Advanced Neural Network Architectures for Multi-Horizon Cryptocurrency Prediction
Includes specialized models for different time horizons and ensemble capabilities
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, BatchNormalization, 
    Attention, MultiHeadAttention, LayerNormalization,
    Bidirectional, Conv1D, MaxPooling1D, GlobalMaxPooling1D,
    Concatenate, Add, Multiply
)
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import RobustScaler, StandardScaler
from typing import Dict, List, Tuple, Optional, Any
import logging

import config

logger = logging.getLogger(__name__)

class BaseTimeSeriesModel:
    """Base class for time series prediction models"""
    
    def __init__(self, input_shape: Tuple[int, int], horizon: int, **kwargs):
        self.input_shape = input_shape
        self.horizon = horizon
        self.model = None
        self.scaler = None
        self.training_history = None
        self.is_trained = False
        
        # Model hyperparameters with stronger regularization
        self.learning_rate = kwargs.get('learning_rate', config.LEARNING_RATE)
        self.batch_size = kwargs.get('batch_size', config.BATCH_SIZE)
        self.epochs = kwargs.get('epochs', config.EPOCHS)
        self.dropout_rate = kwargs.get('dropout_rate', 0.4)  # Increased from 0.2
        self.l1_reg = kwargs.get('l1_reg', 0.001)
        self.l2_reg = kwargs.get('l2_reg', 0.001)
        
    def build_model(self) -> Model:
        """Build the model architecture - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement build_model")
    
    def compile_model(self, model: Model) -> Model:
        """Compile model with appropriate optimizer and loss function"""
        optimizer = Adam(learning_rate=self.learning_rate, clipnorm=1.0)  # Gradient clipping
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mape']
        )
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              validation_data: Tuple[np.ndarray, np.ndarray] = None,
              progress_callback: callable = None) -> Dict:
        """Train the model"""
        if self.model is None:
            self.model = self.build_model()
            self.model = self.compile_model(self.model)
        
        # Callbacks with more aggressive early stopping
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, min_delta=0.001),
            ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-6)
        ]
        
        # Custom callback for progress reporting
        if progress_callback:
            class ProgressCallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    progress_callback(epoch, logs or {})
            callbacks.append(ProgressCallback())
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=0
        )
        
        self.training_history = history.history
        self.is_trained = True
        
        return self.training_history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.model is None or not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X, verbose=0)
    
    def get_model_summary(self) -> str:
        """Get model architecture summary"""
        if self.model is None:
            return "Model not built yet"
        return self.model.summary()


class ShortTermModel(BaseTimeSeriesModel):
    """
    Specialized model for 1-day predictions
    Uses shallow LSTM with attention mechanism for quick market movements
    """
    
    def __init__(self, input_shape: Tuple[int, int], horizon: int = 1, **kwargs):
        super().__init__(input_shape, horizon, **kwargs)
        self.attention_units = kwargs.get('attention_units', 32)
        self.lstm_units = kwargs.get('lstm_units', 64)
    
    def build_model(self) -> Model:
        """Build short-term prediction model"""
        inputs = Input(shape=self.input_shape)
        
        # Shallow LSTM layer
        lstm_out = LSTM(
            self.lstm_units, 
            return_sequences=True,
            dropout=self.dropout_rate
        )(inputs)
        
        # Attention mechanism for focusing on recent patterns
        attention = MultiHeadAttention(
            num_heads=4,
            key_dim=self.attention_units
        )(lstm_out, lstm_out)
        
        # Layer normalization
        attention = LayerNormalization()(attention)
        
        # Global pooling
        pooled = GlobalMaxPooling1D()(attention)
        
        # Dense layers with regularization
        dense1 = Dense(32, activation='relu', 
                      kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg))(pooled)
        dense1 = Dropout(self.dropout_rate)(dense1)
        dense1 = BatchNormalization()(dense1)
        
        dense2 = Dense(16, activation='relu',
                      kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg))(dense1)
        dense2 = Dropout(self.dropout_rate)(dense2)
        
        # Output layer
        outputs = Dense(1, activation='linear')(dense2)
        
        model = Model(inputs=inputs, outputs=outputs, name='ShortTermModel')
        return model


class MediumTermModel(BaseTimeSeriesModel):
    """
    Specialized model for 7-day predictions
    Uses bidirectional LSTM to capture both forward and backward patterns
    """
    
    def __init__(self, input_shape: Tuple[int, int], horizon: int = 7, **kwargs):
        super().__init__(input_shape, horizon, **kwargs)
        self.lstm_units = kwargs.get('lstm_units', 128)
        self.dense_units = kwargs.get('dense_units', 64)
    
    def build_model(self) -> Model:
        """Build medium-term prediction model"""
        inputs = Input(shape=self.input_shape)
        
        # Bidirectional LSTM layers
        bilstm1 = Bidirectional(LSTM(
            self.lstm_units, 
            return_sequences=True,
            dropout=self.dropout_rate
        ))(inputs)
        bilstm1 = BatchNormalization()(bilstm1)
        
        bilstm2 = Bidirectional(LSTM(
            self.lstm_units // 2,
            return_sequences=False,
            dropout=self.dropout_rate
        ))(bilstm1)
        bilstm2 = BatchNormalization()(bilstm2)
        
        # Dense layers with residual connections and regularization
        dense1 = Dense(self.dense_units, activation='relu',
                      kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg))(bilstm2)
        dense1 = Dropout(self.dropout_rate)(dense1)
        dense1 = BatchNormalization()(dense1)
        
        dense2 = Dense(self.dense_units // 2, activation='relu',
                      kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg))(dense1)
        dense2 = Dropout(self.dropout_rate)(dense2)
        
        # Skip connection
        skip = Dense(self.dense_units // 2, activation='linear')(bilstm2)
        combined = Add()([dense2, skip])
        
        # Output layer
        outputs = Dense(1, activation='linear')(combined)
        
        model = Model(inputs=inputs, outputs=outputs, name='MediumTermModel')
        return model


class LongTermModel(BaseTimeSeriesModel):
    """
    Specialized model for 14-30 day predictions
    Uses transformer-like architecture for long-term dependencies
    """
    
    def __init__(self, input_shape: Tuple[int, int], horizon: int = 30, **kwargs):
        super().__init__(input_shape, horizon, **kwargs)
        self.d_model = kwargs.get('d_model', 128)
        self.num_heads = kwargs.get('num_heads', 8)
        self.ff_dim = kwargs.get('ff_dim', 256)
        self.num_transformer_blocks = kwargs.get('num_transformer_blocks', 2)
    
    def transformer_block(self, inputs, d_model, num_heads, ff_dim, dropout_rate=0.1):
        """Transformer block implementation"""
        # Multi-head attention
        attention = MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=d_model
        )(inputs, inputs)
        attention = Dropout(dropout_rate)(attention)
        attention = LayerNormalization()(Add()([inputs, attention]))
        
        # Feed forward network
        ff_output = Dense(ff_dim, activation='relu')(attention)
        ff_output = Dense(d_model)(ff_output)
        ff_output = Dropout(dropout_rate)(ff_output)
        ff_output = LayerNormalization()(Add()([attention, ff_output]))
        
        return ff_output
    
    def build_model(self) -> Model:
        """Build long-term prediction model"""
        inputs = Input(shape=self.input_shape)
        
        # Initial dense projection to d_model dimensions
        x = Dense(self.d_model)(inputs)
        
        # Multiple transformer blocks
        for _ in range(self.num_transformer_blocks):
            x = self.transformer_block(
                x, self.d_model, self.num_heads, 
                self.ff_dim, self.dropout_rate
            )
        
        # Global average pooling
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        # Final dense layers with regularization
        x = Dense(self.ff_dim // 2, activation='relu',
                 kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg))(x)
        x = Dropout(self.dropout_rate)(x)
        x = BatchNormalization()(x)
        
        x = Dense(self.ff_dim // 4, activation='relu',
                 kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg))(x)
        x = Dropout(self.dropout_rate)(x)
        
        # Output layer
        outputs = Dense(1, activation='linear')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='LongTermModel')
        return model


class HybridCNNLSTMModel(BaseTimeSeriesModel):
    """
    Hybrid model combining CNN for pattern recognition and LSTM for sequences
    Good for medium-term predictions with complex patterns
    """
    
    def __init__(self, input_shape: Tuple[int, int], horizon: int = 14, **kwargs):
        super().__init__(input_shape, horizon, **kwargs)
        self.conv_filters = kwargs.get('conv_filters', [64, 32])
        self.conv_kernel_size = kwargs.get('conv_kernel_size', 3)
        self.lstm_units = kwargs.get('lstm_units', 128)
    
    def build_model(self) -> Model:
        """Build hybrid CNN-LSTM model"""
        inputs = Input(shape=self.input_shape)
        
        # CNN layers for pattern recognition
        x = inputs
        for filters in self.conv_filters:
            x = Conv1D(
                filters, 
                self.conv_kernel_size, 
                activation='relu',
                padding='same'
            )(x)
            x = BatchNormalization()(x)
            x = Dropout(self.dropout_rate)(x)
        
        # Max pooling to reduce dimensionality
        x = MaxPooling1D(pool_size=2)(x)
        
        # LSTM layers for sequence modeling
        x = LSTM(
            self.lstm_units, 
            return_sequences=True,
            dropout=self.dropout_rate
        )(x)
        x = BatchNormalization()(x)
        
        x = LSTM(
            self.lstm_units // 2,
            return_sequences=False,
            dropout=self.dropout_rate
        )(x)
        x = BatchNormalization()(x)
        
        # Dense layers with regularization
        x = Dense(64, activation='relu',
                 kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg))(x)
        x = Dropout(self.dropout_rate)(x)
        
        x = Dense(32, activation='relu',
                 kernel_regularizer=l1_l2(l1=self.l1_reg, l2=self.l2_reg))(x)
        x = Dropout(self.dropout_rate)(x)
        
        # Output layer
        outputs = Dense(1, activation='linear')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='HybridCNNLSTMModel')
        return model


class EnsembleModel:
    """
    Ensemble model that combines predictions from multiple architectures
    """
    
    def __init__(self, input_shape: Tuple[int, int], horizon: int, **kwargs):
        self.input_shape = input_shape
        self.horizon = horizon
        self.models = {}
        self.weights = {}
        self.is_trained = False
        
        # Initialize different model architectures
        self.model_configs = {
            'short_term': {'class': ShortTermModel, 'weight': 0.3},
            'medium_term': {'class': MediumTermModel, 'weight': 0.4},
            'hybrid': {'class': HybridCNNLSTMModel, 'weight': 0.3}
        }
        
        # Adjust weights based on horizon
        if horizon == 1:
            self.model_configs['short_term']['weight'] = 0.6
            self.model_configs['medium_term']['weight'] = 0.3
            self.model_configs['hybrid']['weight'] = 0.1
        elif horizon <= 7:
            self.model_configs['short_term']['weight'] = 0.2
            self.model_configs['medium_term']['weight'] = 0.6
            self.model_configs['hybrid']['weight'] = 0.2
        else:
            self.model_configs['short_term']['weight'] = 0.1
            self.model_configs['medium_term']['weight'] = 0.3
            self.model_configs['hybrid']['weight'] = 0.6
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              validation_data: Tuple[np.ndarray, np.ndarray] = None,
              progress_callback: callable = None) -> Dict:
        """Train all ensemble models"""
        training_results = {}
        total_models = len(self.model_configs)
        
        for i, (name, config) in enumerate(self.model_configs.items()):
            logger.info(f"Training {name} model ({i+1}/{total_models})")
            
            # Initialize model
            model_class = config['class']
            model = model_class(self.input_shape, self.horizon)
            
            # Custom progress callback for ensemble training
            def ensemble_progress_callback(epoch, logs):
                if progress_callback:
                    overall_progress = (i / total_models) + (epoch / config.get('epochs', 50)) * (1 / total_models)
                    progress_callback(epoch, {
                        **logs,
                        'ensemble_progress': overall_progress,
                        'current_model': name
                    })
            
            # Train model
            history = model.train(
                X_train, y_train,
                validation_data=validation_data,
                progress_callback=ensemble_progress_callback
            )
            
            # Store model and results
            self.models[name] = model
            self.weights[name] = config['weight']
            training_results[name] = history
        
        self.is_trained = True
        return training_results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions"""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")
        
        predictions = []
        weights = []
        
        for name, model in self.models.items():
            pred = model.predict(X)
            predictions.append(pred)
            weights.append(self.weights[name])
        
        # Weighted average of predictions
        weighted_preds = np.average(predictions, axis=0, weights=weights)
        return weighted_preds
    
    def get_individual_predictions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get predictions from individual models"""
        individual_preds = {}
        
        for name, model in self.models.items():
            individual_preds[name] = model.predict(X)
        
        return individual_preds
    
    def get_confidence_intervals(self, X: np.ndarray, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate confidence intervals based on model disagreement
        
        Args:
            X: Input data
            alpha: Significance level (0.05 for 95% confidence)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        individual_preds = self.get_individual_predictions(X)
        pred_array = np.array(list(individual_preds.values()))
        
        # Calculate percentiles based on model predictions
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(pred_array, lower_percentile, axis=0)
        upper_bound = np.percentile(pred_array, upper_percentile, axis=0)
        
        return lower_bound, upper_bound


def get_model_for_horizon(horizon: int, input_shape: Tuple[int, int], 
                         use_ensemble: bool = True, **kwargs) -> BaseTimeSeriesModel:
    """
    Factory function to get appropriate model for prediction horizon
    
    Args:
        horizon: Prediction horizon in days
        input_shape: Input shape (sequence_length, features)
        use_ensemble: Whether to use ensemble model
        **kwargs: Additional model parameters
        
    Returns:
        Appropriate model instance
    """
    if use_ensemble:
        return EnsembleModel(input_shape, horizon, **kwargs)
    
    # Single model selection based on horizon
    if horizon == 1:
        return ShortTermModel(input_shape, horizon, **kwargs)
    elif horizon <= 7:
        return MediumTermModel(input_shape, horizon, **kwargs)
    elif horizon <= 14:
        return HybridCNNLSTMModel(input_shape, horizon, **kwargs)
    else:
        return LongTermModel(input_shape, horizon, **kwargs)


# Model registry for easy access
MODEL_REGISTRY = {
    'short_term': ShortTermModel,
    'medium_term': MediumTermModel,
    'long_term': LongTermModel,
    'hybrid_cnn_lstm': HybridCNNLSTMModel,
    'ensemble': EnsembleModel
}