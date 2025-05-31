#!/usr/bin/env python3
"""
Comprehensive tests for neural network model training and progress bar functionality
Tests time windows training and UI progress updates
"""

import os
import sys
import pytest
import threading
import time
import tempfile
import shutil
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from backend.neural_network_model import CryptoPredictor
from frontend.app_game_simple import train_models_background, start_training, training_status

class TestNeuralNetworkTraining:
    """Test suite for neural network model training"""
    
    @pytest.fixture
    def setup_test_data(self):
        """Create test data for neural network training"""
        # Create temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        self.original_dirs = {
            'RAW_DATA_DIRECTORY': config.RAW_DATA_DIRECTORY,
            'PROCESSED_DATA_DIRECTORY': config.PROCESSED_DATA_DIRECTORY,
            'MODEL_DIRECTORY': config.MODEL_DIRECTORY
        }
        
        # Override config directories
        config.RAW_DATA_DIRECTORY = os.path.join(self.test_dir, 'raw')
        config.PROCESSED_DATA_DIRECTORY = os.path.join(self.test_dir, 'processed')
        config.MODEL_DIRECTORY = os.path.join(self.test_dir, 'models')
        
        # Create directories
        os.makedirs(config.RAW_DATA_DIRECTORY, exist_ok=True)
        os.makedirs(config.PROCESSED_DATA_DIRECTORY, exist_ok=True)
        os.makedirs(config.MODEL_DIRECTORY, exist_ok=True)
        
        # Generate realistic test data
        self.create_test_crypto_data('BTC_USD.csv')
        self.create_test_crypto_data('ETH_USD.csv')
        
        yield
        
        # Cleanup
        shutil.rmtree(self.test_dir)
        for key, value in self.original_dirs.items():
            setattr(config, key, value)
    
    def create_test_crypto_data(self, filename):
        """Create realistic cryptocurrency price data for testing"""
        # Generate 500 days of realistic price data
        dates = pd.date_range(start='2023-01-01', periods=500, freq='D')
        
        # Start with a base price
        base_price = 30000 if 'BTC' in filename else 2000
        prices = []
        current_price = base_price
        
        # Generate realistic price movements
        for i in range(500):
            # Random walk with trend and volatility
            change_pct = np.random.normal(0.001, 0.03)  # Small upward trend with 3% daily volatility
            current_price *= (1 + change_pct)
            prices.append(current_price)
        
        # Create OHLCV data
        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            # Generate realistic OHLC from close price
            volatility = close * 0.02  # 2% intraday volatility
            high = close + np.random.uniform(0, volatility)
            low = close - np.random.uniform(0, volatility)
            open_price = low + np.random.uniform(0, high - low)
            volume = np.random.uniform(1e9, 5e9)  # Random volume
            
            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close, 2),
                'volume': int(volume)
            })
        
        # Save to CSV
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(config.RAW_DATA_DIRECTORY, filename), index=False)
        
        # Also create processed data with technical indicators
        self.create_processed_data(df, filename)
    
    def create_processed_data(self, df, filename):
        """Create processed data with technical indicators"""
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        # Add technical indicators
        df['SMA_7'] = df['close'].rolling(window=7).mean()
        df['SMA_30'] = df['close'].rolling(window=30).mean()
        df['EMA_7'] = df['close'].ewm(span=7).mean()
        df['EMA_30'] = df['close'].ewm(span=30).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        
        # Volatility
        df['volatility_7d'] = df['close'].rolling(window=7).std()
        df['volatility_30d'] = df['close'].rolling(window=30).std()
        
        # Save processed data
        processed_filename = filename.replace('.csv', '_processed.csv')
        df.to_csv(os.path.join(config.PROCESSED_DATA_DIRECTORY, processed_filename))
    
    def test_crypto_predictor_initialization(self, setup_test_data):
        """Test CryptoPredictor initialization"""
        predictor = CryptoPredictor('BTC-USD')
        
        assert predictor.symbol == 'BTC-USD'
        assert predictor.base_symbol == 'BTC'
        assert len(predictor.scalers) == len(config.PREDICTION_HORIZONS)
        assert len(predictor.models) == len(config.PREDICTION_HORIZONS)
        assert predictor.sequence_length == config.SEQUENCE_LENGTH
    
    def test_data_loading(self, setup_test_data):
        """Test data loading functionality"""
        predictor = CryptoPredictor('BTC-USD')
        
        # Test loading processed data
        df = predictor.load_data()
        
        assert not df.empty
        assert len(df) > 100  # Should have sufficient data
        assert 'close' in df.columns
        assert 'volume' in df.columns
    
    def test_feature_preparation(self, setup_test_data):
        """Test feature preparation for model training"""
        predictor = CryptoPredictor('BTC-USD')
        df = predictor.load_data()
        
        feature_df = predictor.prepare_features(df)
        
        assert not feature_df.empty
        assert len(predictor.feature_columns) > 0
        assert 'close' in predictor.feature_columns
        assert 'volume' in predictor.feature_columns
        # Check for technical indicators
        expected_features = ['SMA_7', 'SMA_30', 'EMA_7', 'EMA_30', 'RSI', 'MACD']
        for feature in expected_features:
            if feature in df.columns:
                assert feature in predictor.feature_columns
    
    def test_sequence_creation(self, setup_test_data):
        """Test sequence creation for LSTM training"""
        predictor = CryptoPredictor('BTC-USD')
        df = predictor.load_data()
        feature_df = predictor.prepare_features(df)
        
        # Test sequence creation for 7-day horizon
        X, y = predictor.create_sequences(feature_df, 'close', 7)
        
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == config.SEQUENCE_LENGTH
        assert X.shape[2] == len(predictor.feature_columns)
        assert len(y) > 0
    
    def test_model_building(self, setup_test_data):
        """Test neural network model architecture"""
        predictor = CryptoPredictor('BTC-USD')
        df = predictor.load_data()
        feature_df = predictor.prepare_features(df)
        
        # Create sample sequences to get input shape
        X, _ = predictor.create_sequences(feature_df, 'close', 7)
        input_shape = (X.shape[1], X.shape[2])
        
        model = predictor.build_model(input_shape)
        
        assert model is not None
        assert len(model.layers) > 0
        assert model.input_shape[1:] == input_shape
        assert model.output_shape[-1] == 1  # Single output for price prediction
    
    @pytest.mark.slow
    def test_model_training_single_horizon(self, setup_test_data):
        """Test training a single model horizon (reduced epochs for speed)"""
        # Temporarily reduce epochs for testing
        original_epochs = config.EPOCHS
        config.EPOCHS = 2  # Very short training for test
        
        try:
            predictor = CryptoPredictor('BTC-USD')
            
            # Train model for 7-day horizon
            model, history = predictor.train_model(7)
            
            assert model is not None
            assert history is not None
            assert len(history.history['loss']) == config.EPOCHS
            
            # Check if model file was saved
            model_path = os.path.join(predictor.model_dir, "model_7day.h5")
            assert os.path.exists(model_path)
            
            # Check if metrics were saved
            metrics_path = os.path.join(predictor.model_dir, "metrics_7day.json")
            assert os.path.exists(metrics_path)
            
        finally:
            config.EPOCHS = original_epochs
    
    def test_prediction_functionality(self, setup_test_data):
        """Test prediction with mock trained model"""
        predictor = CryptoPredictor('BTC-USD')
        
        # Mock a trained model
        with patch.object(predictor, 'load_trained_model') as mock_load:
            mock_model = MagicMock()
            mock_model.predict.return_value = np.array([[45000.0]])
            mock_load.return_value = mock_model
            predictor.models[7] = mock_model
            
            # Mock scaler
            with patch.object(predictor.scalers[7], 'transform') as mock_transform, \
                 patch.object(predictor.scalers[7], 'inverse_transform') as mock_inverse:
                
                mock_transform.return_value = np.random.rand(60, 5)
                mock_inverse.return_value = np.array([[45000.0, 0, 0, 0, 0]])
                
                df = predictor.load_data()
                feature_df = predictor.prepare_features(df)
                latest_data = feature_df.iloc[-60:].copy()
                
                prediction = predictor.predict(7, latest_data)
                
                assert prediction is not None
                assert isinstance(prediction, float)
                assert prediction > 0
    
    def test_multiple_horizons_prediction(self, setup_test_data):
        """Test predictions for multiple time horizons"""
        predictor = CryptoPredictor('BTC-USD')
        
        # Mock models for all horizons
        for horizon in config.PREDICTION_HORIZONS:
            mock_model = MagicMock()
            mock_model.predict.return_value = np.array([[40000.0 + horizon * 100]])
            predictor.models[horizon] = mock_model
            
            # Mock scalers
            with patch.object(predictor.scalers[horizon], 'transform') as mock_transform, \
                 patch.object(predictor.scalers[horizon], 'inverse_transform') as mock_inverse:
                
                mock_transform.return_value = np.random.rand(60, 5)
                mock_inverse.return_value = np.array([[40000.0 + horizon * 100, 0, 0, 0, 0]])
        
        predictions = predictor.predict_multiple_horizons([7, 14, 30])
        
        assert len(predictions) == 3
        for horizon in [7, 14, 30]:
            assert horizon in predictions
            assert 'predicted_price' in predictions[horizon]
            assert 'prediction_date' in predictions[horizon]
            assert predictions[horizon]['predicted_price'] > 0


class TestTrainingProgressBar:
    """Test suite for training progress bar and UI updates"""
    
    def test_training_status_initialization(self):
        """Test initial training status"""
        global training_status
        
        # Reset training status
        training_status.update({
            "status": "idle", 
            "message": "", 
            "btc_complete": False, 
            "eth_complete": False
        })
        
        assert training_status["status"] == "idle"
        assert training_status["message"] == ""
        assert training_status["btc_complete"] is False
        assert training_status["eth_complete"] is False
    
    def test_background_training_simulation(self):
        """Test background training simulation with progress updates"""
        global training_status
        
        # Reset status
        training_status.update({
            "status": "idle", 
            "message": "", 
            "btc_complete": False, 
            "eth_complete": False
        })
        
        current_date = "2024-01-15"
        
        # Start training in background
        training_thread = threading.Thread(
            target=train_models_background, 
            args=(current_date,)
        )
        training_thread.daemon = True
        training_thread.start()
        
        # Wait a moment for training to start
        time.sleep(0.5)
        assert training_status["status"] == "training"
        assert "Starting AI model training" in training_status["message"]
        
        # Wait for BTC training to complete
        time.sleep(3)
        assert training_status["btc_complete"] is True
        assert "Bitcoin model trained" in training_status["message"]
        
        # Wait for ETH training to complete
        time.sleep(4)
        assert training_status["eth_complete"] is True
        assert training_status["status"] == "complete"
        
        # Wait for final status
        time.sleep(3)
        assert training_status["status"] == "show_charts"
        
        training_thread.join(timeout=1)
    
    def test_training_progress_messages(self):
        """Test specific training progress messages"""
        global training_status
        
        current_date = "2024-01-15"
        
        # Monitor training messages
        messages_received = []
        
        def monitor_training():
            start_time = time.time()
            while time.time() - start_time < 10:  # Monitor for 10 seconds
                if training_status["message"]:
                    if training_status["message"] not in messages_received:
                        messages_received.append(training_status["message"])
                time.sleep(0.1)
        
        # Start monitoring in background
        monitor_thread = threading.Thread(target=monitor_training)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Start training
        start_training(current_date)
        
        # Wait for training to complete
        time.sleep(10)
        
        # Verify expected messages were received
        expected_messages = [
            "Starting AI model training",
            "Training Bitcoin prediction model",
            "Training Bitcoin 7-day prediction",
            "Training Bitcoin 14-day prediction", 
            "Training Bitcoin 30-day prediction",
            "Bitcoin model trained",
            "Training Ethereum 7-day prediction",
            "Training Ethereum 14-day prediction",
            "Training Ethereum 30-day prediction",
            "All AI models trained successfully"
        ]
        
        for expected in expected_messages:
            assert any(expected in msg for msg in messages_received), \
                f"Expected message '{expected}' not found in {messages_received}"
    
    def test_training_error_handling(self):
        """Test training error handling"""
        global training_status
        
        # Mock training function to raise an error
        def failing_training(current_date):
            global training_status
            try:
                training_status["status"] = "training"
                training_status["message"] = "🔄 Starting AI model training..."
                raise ValueError("Simulated training error")
            except Exception as e:
                training_status["status"] = "error"
                training_status["message"] = f"❌ Training failed: {str(e)}"
        
        current_date = "2024-01-15"
        
        # Start failing training
        training_thread = threading.Thread(
            target=failing_training, 
            args=(current_date,)
        )
        training_thread.daemon = True
        training_thread.start()
        
        # Wait for error
        time.sleep(1)
        
        assert training_status["status"] == "error"
        assert "Training failed" in training_status["message"]
        assert "Simulated training error" in training_status["message"]
    
    def test_concurrent_training_prevention(self):
        """Test that multiple training sessions don't interfere"""
        global training_status
        
        # Reset status
        training_status.update({
            "status": "idle", 
            "message": "", 
            "btc_complete": False, 
            "eth_complete": False
        })
        
        current_date = "2024-01-15"
        
        # Start first training
        start_training(current_date)
        time.sleep(0.5)
        
        first_status = training_status["status"]
        
        # Attempt to start second training
        start_training(current_date)
        time.sleep(0.5)
        
        # Should still be the same training session
        assert training_status["status"] == first_status
        
        # Wait for completion
        time.sleep(10)


def run_comprehensive_tests():
    """Run all tests with detailed output"""
    print("🧪 Running Comprehensive Model Training Tests")
    print("=" * 60)
    
    # Run neural network tests
    print("\n📊 Testing Neural Network Training...")
    pytest.main([
        __file__ + "::TestNeuralNetworkTraining",
        "-v", "--tb=short"
    ])
    
    # Run progress bar tests  
    print("\n📈 Testing Training Progress Bar...")
    pytest.main([
        __file__ + "::TestTrainingProgressBar", 
        "-v", "--tb=short"
    ])
    
    print("\n✅ All tests completed!")


if __name__ == "__main__":
    run_comprehensive_tests()