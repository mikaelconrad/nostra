"""
Pytest configuration and fixtures for the cryptocurrency investment app
"""

import pytest
import os
import sys
import tempfile
import shutil
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from backend.portfolio_tracker import PortfolioTracker
from api.app import app as flask_app

@pytest.fixture
def app():
    """Create and configure a test Flask application"""
    flask_app.config.update({
        "TESTING": True,
    })
    yield flask_app

@pytest.fixture
def client(app):
    """Create a test client for the Flask application"""
    return app.test_client()

@pytest.fixture
def runner(app):
    """Create a test CLI runner for the Flask application"""
    return app.test_cli_runner()

@pytest.fixture
def temp_data_dir():
    """Create a temporary data directory for testing"""
    temp_dir = tempfile.mkdtemp()
    
    # Create subdirectories
    os.makedirs(os.path.join(temp_dir, 'raw'), exist_ok=True)
    os.makedirs(os.path.join(temp_dir, 'processed'), exist_ok=True)
    os.makedirs(os.path.join(temp_dir, 'sentiment'), exist_ok=True)
    os.makedirs(os.path.join(temp_dir, 'recommendations'), exist_ok=True)
    
    # Override config data directory
    original_data_dir = config.DATA_DIRECTORY
    config.DATA_DIRECTORY = temp_dir
    config.RAW_DATA_DIRECTORY = os.path.join(temp_dir, 'raw')
    config.PROCESSED_DATA_DIRECTORY = os.path.join(temp_dir, 'processed')
    config.SENTIMENT_DATA_DIRECTORY = os.path.join(temp_dir, 'sentiment')
    config.RECOMMENDATIONS_DIRECTORY = os.path.join(temp_dir, 'recommendations')
    config.PORTFOLIO_FILE = os.path.join(temp_dir, 'portfolio.json')
    config.TRANSACTION_HISTORY_FILE = os.path.join(temp_dir, 'transactions.json')
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)
    
    # Restore original config
    config.DATA_DIRECTORY = original_data_dir
    config.RAW_DATA_DIRECTORY = os.path.join(original_data_dir, 'raw')
    config.PROCESSED_DATA_DIRECTORY = os.path.join(original_data_dir, 'processed')
    config.SENTIMENT_DATA_DIRECTORY = os.path.join(original_data_dir, 'sentiment')
    config.RECOMMENDATIONS_DIRECTORY = os.path.join(original_data_dir, 'recommendations')
    config.PORTFOLIO_FILE = os.path.join(original_data_dir, 'portfolio.json')
    config.TRANSACTION_HISTORY_FILE = os.path.join(original_data_dir, 'transactions.json')

@pytest.fixture
def portfolio_tracker(temp_data_dir):
    """Create a portfolio tracker instance with temporary data directory"""
    return PortfolioTracker()

@pytest.fixture
def sample_portfolio_data():
    """Sample portfolio data for testing"""
    return {
        'cash': 5000.0,
        'holdings': {
            'BTC': 0.1,
            'ETH': 2.5
        },
        'last_update': datetime.now().isoformat()
    }

@pytest.fixture
def sample_transaction_data():
    """Sample transaction data for testing"""
    return [
        {
            'type': 'buy',
            'symbol': 'BTC',
            'amount': 0.1,
            'price': 50000.0,
            'total': 5000.0,
            'timestamp': '2024-01-01T10:00:00'
        },
        {
            'type': 'buy',
            'symbol': 'ETH',
            'amount': 2.5,
            'price': 2000.0,
            'total': 5000.0,
            'timestamp': '2024-01-02T10:00:00'
        }
    ]

@pytest.fixture
def sample_price_data():
    """Sample price data for testing"""
    import pandas as pd
    
    dates = pd.date_range(end=datetime.now(), periods=30)
    
    return {
        'BTC': pd.DataFrame({
            'date': dates.strftime('%Y-%m-%d'),
            'open': [50000 + i*100 for i in range(30)],
            'high': [50500 + i*100 for i in range(30)],
            'low': [49500 + i*100 for i in range(30)],
            'close': [50000 + i*100 for i in range(30)],
            'volume': [1000000 + i*1000 for i in range(30)]
        }),
        'ETH': pd.DataFrame({
            'date': dates.strftime('%Y-%m-%d'),
            'open': [2000 + i*10 for i in range(30)],
            'high': [2050 + i*10 for i in range(30)],
            'low': [1950 + i*10 for i in range(30)],
            'close': [2000 + i*10 for i in range(30)],
            'volume': [500000 + i*500 for i in range(30)]
        }),
        'XRP': pd.DataFrame({
            'date': dates.strftime('%Y-%m-%d'),
            'open': [0.5 + i*0.01 for i in range(30)],
            'high': [0.52 + i*0.01 for i in range(30)],
            'low': [0.48 + i*0.01 for i in range(30)],
            'close': [0.5 + i*0.01 for i in range(30)],
            'volume': [100000 + i*100 for i in range(30)]
        })
    }

@pytest.fixture
def mock_current_prices():
    """Mock current prices for testing"""
    return {
        'BTC': 52900.0,
        'ETH': 2290.0,
        'XRP': 0.79
    }