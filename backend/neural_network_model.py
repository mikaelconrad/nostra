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
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json

# Import configuration
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class CryptoPredictor:
    def __init__(self, symbol):
        """
        Initialize the cryptocurrency predictor
        
        Args:
            symbol: Cryptocurrency symbol (e.g., 'BTC-USD')
        """
        self.symbol = symbol
        self.base_symbol = symbol.split('-')[0]
        self.model_dir = os.path.join(config.MODEL_DIRECTORY, self.base_symbol)
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize scalers for each prediction horizon
        self.scalers = {horizon: MinMaxScaler() for horizon in config.PREDICTION_HORIZONS}
        
        # Initialize models for each prediction horizon
        self.models = {horizon: None for horizon in config.PREDICTION_HORIZONS}
        
        # Initialize feature columns
        self.feature_columns = []
        
        # Initialize sequence length
        self.sequence_length = config.SEQUENCE_LENGTH
        
    def load_data(self):
        """
        Load processed cryptocurrency data and sentiment data
        
        Returns:
            DataFrame with combined price and sentiment data
        """
        # Load price data
        price_file = os.path.join(config.DATA_DIRECTORY, 'processed', f"{self.symbol.replace('-', '_')}_processed.csv")
        if not os.path.exists(price_file):
            raise FileNotFoundError(f"Price data file not found: {price_file}")
        
        price_df = pd.read_csv(price_file, index_col='date', parse_dates=True)
        print(f"Loaded price data for {self.symbol}: {len(price_df)} records")
        
        # Try to load sentiment data
        sentiment_file = os.path.join(config.SENTIMENT_DATA_DIRECTORY, f"{self.base_symbol}_daily_sentiment.csv")
        try:
            if os.path.exists(sentiment_file):
                sentiment_df = pd.read_csv(sentiment_file, parse_dates=['created_at'])
                sentiment_df = sentiment_df.set_index('created_at')
                
                # Merge price and sentiment data
                merged_df = price_df.join(sentiment_df, how='left')
                
                # Fill missing sentiment values with neutral values
                if 'vader_compound' in merged_df.columns:
                    merged_df['vader_compound'].fillna(0, inplace=True)
                if 'textblob_polarity' in merged_df.columns:
                    merged_df['textblob_polarity'].fillna(0, inplace=True)
                
                print(f"Merged sentiment data: {len(merged_df)} records")
                return merged_df
            else:
                print(f"Sentiment data file not found: {sentiment_file}")
                return price_df
        except Exception as e:
            print(f"Error loading sentiment data: {str(e)}")
            return price_df
    
    def prepare_features(self, df):
        """
        Prepare features for the model
        
        Args:
            df: DataFrame with price and sentiment data
            
        Returns:
            DataFrame with selected features
        """
        # Select relevant columns for features
        feature_df = df.copy()
        
        # Define feature columns based on available data
        self.feature_columns = []
        
        # Always include price and volume data
        price_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in price_columns:
            if col in feature_df.columns:
                self.feature_columns.append(col)
        
        # Include technical indicators if available
        tech_indicators = ['SMA_7', 'SMA_30', 'EMA_7', 'EMA_30', 'RSI', 'MACD', 
                          'volatility_7d', 'volatility_30d']
        for col in tech_indicators:
            if col in feature_df.columns:
                self.feature_columns.append(col)
        
        # Include sentiment data if available
        sentiment_columns = ['vader_compound', 'textblob_polarity']
        for col in sentiment_columns:
            if col in feature_df.columns:
                self.feature_columns.append(col)
        
        print(f"Selected features: {self.feature_columns}")
        
        # Drop rows with NaN values
        feature_df = feature_df[self.feature_columns].dropna()
        
        return feature_df
    
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
        Build LSTM model for sequence prediction
        
        Args:
            input_shape: Shape of input sequences
            
        Returns:
            Compiled Keras model
        """
        model = Sequential()
        
        # LSTM layers
        model.add(LSTM(units=128, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        
        model.add(LSTM(units=64, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        
        # Dense layers
        model.add(Dense(units=32, activation='relu'))
        model.add(Dropout(0.2))
        
        # Output layer
        model.add(Dense(units=1))
        
        # Compile model
        optimizer = Adam(learning_rate=config.LEARNING_RATE)
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        
        return model
    
    def train_model(self, horizon):
        """
        Train model for a specific prediction horizon
        
        Args:
            horizon: Number of days ahead to predict
            
        Returns:
            Trained model and training history
        """
        print(f"Training model for {horizon}-day prediction horizon")
        
        # Load and prepare data
        df = self.load_data()
        feature_df = self.prepare_features(df)
        
        # Scale features
        scaled_data = self.scalers[horizon].fit_transform(feature_df)
        scaled_df = pd.DataFrame(scaled_data, columns=feature_df.columns, index=feature_df.index)
        
        # Create sequences
        X, y = self.create_sequences(scaled_df, 'close', horizon)
        
        # Split into training and testing sets
        split_idx = int(len(X) * config.TRAIN_TEST_SPLIT)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}")
        
        # Build model
        model = self.build_model((X_train.shape[1], X_train.shape[2]))
        
        # Define callbacks
        model_path = os.path.join(self.model_dir, f"model_{horizon}day.h5")
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint(filepath=model_path, save_best_only=True)
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model
        test_loss = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test loss: {test_loss}")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"MSE: {mse}, MAE: {mae}, R²: {r2}")
        
        # Save metrics
        metrics = {
            'mse': float(mse),
            'mae': float(mae),
            'r2': float(r2),
            'test_loss': float(test_loss)
        }
        
        metrics_path = os.path.join(self.model_dir, f"metrics_{horizon}day.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Save model
        self.models[horizon] = model
        
        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'{self.symbol} {horizon}-Day Prediction Model Training')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(self.model_dir, f"training_history_{horizon}day.png"))
        
        return model, history
    
    def load_trained_model(self, horizon):
        """
        Load a trained model for a specific prediction horizon
        
        Args:
            horizon: Number of days ahead to predict
            
        Returns:
            Loaded model
        """
        model_path = os.path.join(self.model_dir, f"model_{horizon}day.h5")
        if os.path.exists(model_path):
            model = load_model(model_path)
            self.models[horizon] = model
            print(f"Loaded model for {horizon}-day prediction horizon")
            return model
        else:
            print(f"Model file not found: {model_path}")
            return None
    
    def predict(self, horizon, latest_data=None):
        """
        Make predictions for a specific horizon
        
        Args:
            horizon: Number of days ahead to predict
            latest_data: Optional latest data to use for prediction
            
        Returns:
            Predicted price
        """
        # Load model if not already loaded
        if self.models[horizon] is None:
            self.load_trained_model(horizon)
            if self.models[horizon] is None:
                print(f"No trained model available for {horizon}-day prediction")
                return None
        
        # Load data if not provided
        if latest_data is None:
            df = self.load_data()
            feature_df = self.prepare_features(df)
            
            # Get the latest sequence
            latest_data = feature_df.iloc[-self.sequence_length:].copy()
        
        # Scale data
        scaled_data = self.scalers[horizon].transform(latest_data)
        
        # Reshape for prediction
        X = np.array([scaled_data])
        
        # Make prediction
        scaled_prediction = self.models[horizon].predict(X)[0][0]
        
        # Inverse transform to get actual price
        # Create a dummy array with zeros except for the close price
        dummy = np.zeros((1, len(self.feature_columns)))
        close_idx = self.feature_columns.index('close')
        dummy[0, close_idx] = scaled_prediction
        
        # Inverse transform
        dummy_inverse = self.scalers[horizon].inverse_transform(dummy)
        prediction = dummy_inverse[0, close_idx]
        
        return prediction
    
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

def train_all_models():
    """
    Train models for all cryptocurrencies and prediction horizons
    """
    for symbol in config.CRYPTO_SYMBOLS:
        predictor = CryptoPredictor(symbol)
        
        for horizon in config.PREDICTION_HORIZONS:
            predictor.train_model(horizon)

def generate_all_recommendations():
    """
    Generate recommendations for all cryptocurrencies
    
    Returns:
        Dictionary with recommendations for all cryptocurrencies
    """
    all_recommendations = {}
    
    for symbol in config.CRYPTO_SYMBOLS:
        predictor = CryptoPredictor(symbol)
        recommendations = predictor.generate_recommendations()
        predictor.save_recommendations(recommendations)
        
        all_recommendations[symbol] = recommendations
    
    return all_recommendations

if __name__ == "__main__":
    # Create model directory
    os.makedirs(config.MODEL_DIRECTORY, exist_ok=True)
    
    # Train all models
    train_all_models()
    
    # Generate recommenda<response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>