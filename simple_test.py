#!/usr/bin/env python3
"""
Simple test to verify core functionality without pytest dependencies
"""
import os
import sys
import time
import threading

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that we can import required modules"""
    print("🔍 Testing imports...")
    
    try:
        import config
        print("✅ config imported successfully")
    except Exception as e:
        print(f"❌ Failed to import config: {e}")
        return False
        
    try:
        from backend.neural_network_model import CryptoPredictor
        print("✅ CryptoPredictor imported successfully")
    except Exception as e:
        print(f"❌ Failed to import CryptoPredictor: {e}")
        return False
        
    try:
        from frontend.app_game_simple import train_models_background, start_training, training_status
        print("✅ Training functions imported successfully")
    except Exception as e:
        print(f"❌ Failed to import training functions: {e}")
        return False
    
    return True

def test_predictor_init():
    """Test CryptoPredictor initialization"""
    print("\n🔍 Testing CryptoPredictor initialization...")
    
    try:
        from backend.neural_network_model import CryptoPredictor
        import config
        
        predictor = CryptoPredictor('BTC-USD')
        
        assert predictor.symbol == 'BTC-USD'
        assert predictor.base_symbol == 'BTC'
        assert len(predictor.scalers) == len(config.PREDICTION_HORIZONS)
        assert len(predictor.models) == len(config.PREDICTION_HORIZONS)
        assert predictor.sequence_length == config.SEQUENCE_LENGTH
        
        print("✅ CryptoPredictor initialization test passed")
        return True
        
    except Exception as e:
        print(f"❌ CryptoPredictor initialization test failed: {e}")
        return False

def test_training_status():
    """Test training status functionality"""
    print("\n🔍 Testing training status...")
    
    try:
        from frontend.app_game_simple import training_status
        
        # Check initial status
        assert "status" in training_status
        assert "message" in training_status  
        assert "btc_complete" in training_status
        assert "eth_complete" in training_status
        
        print("✅ Training status structure test passed")
        return True
        
    except Exception as e:
        print(f"❌ Training status test failed: {e}")
        return False

def test_background_training():
    """Test background training simulation"""
    print("\n🔍 Testing background training...")
    
    try:
        from frontend.app_game_simple import train_models_background, training_status
        
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
        
        print("✅ Background training simulation test passed")
        return True
        
    except Exception as e:
        print(f"❌ Background training test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Running Simple Model Training Verification Tests")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_predictor_init,
        test_training_status,
        test_background_training
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} encountered an error: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! Model training verification successful.")
        return 0
    else:
        print(f"❌ {total - passed} tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)