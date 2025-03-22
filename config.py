"""
Configuration file for Cryptocurrency Investment Recommendation System
"""

import os

# Application settings
DASHBOARD_TITLE = "Cryptocurrency Investment Recommendation System"
FRONTEND_PORT = 8050
THEME = 'light'  # Add this line

# Investment settings
INITIAL_INVESTMENT = 1000  # CHF
MONTHLY_CONTRIBUTION = 500  # CHF

# Cryptocurrency settings
CRYPTO_SYMBOLS = {
    'BTC-USD': 'Bitcoin',
    'ETH-USD': 'Ethereum',
    'XRP-USD': 'Ripple'
}

# Prediction settings
PREDICTION_HORIZONS = [1, 7, 30]  # days

# Directory settings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIRECTORY = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIRECTORY = os.path.join(DATA_DIRECTORY, 'raw')  # Make sure this line exists
PROCESSED_DATA_DIRECTORY = os.path.join(DATA_DIRECTORY, 'processed')
SENTIMENT_DATA_DIRECTORY = os.path.join(DATA_DIRECTORY, 'sentiment')
RECOMMENDATIONS_DIRECTORY = os.path.join(DATA_DIRECTORY, 'recommendations')
MODEL_DIRECTORY = os.path.join(BASE_DIR, 'models')
REPORT_DIRECTORY = os.path.join(BASE_DIR, 'reports')

# File settings
PORTFOLIO_FILE = os.path.join(DATA_DIRECTORY, 'portfolio.json')
TRANSACTION_HISTORY_FILE = os.path.join(DATA_DIRECTORY, 'transactions.json')

# Create directories if they don't exist
for directory in [DATA_DIRECTORY, RAW_DATA_DIRECTORY, PROCESSED_DATA_DIRECTORY, 
                 SENTIMENT_DATA_DIRECTORY, RECOMMENDATIONS_DIRECTORY, 
                 MODEL_DIRECTORY, REPORT_DIRECTORY]:
    os.makedirs(directory, exist_ok=True)

# Neural network settings
SEQUENCE_LENGTH = 60  # days of historical data to use for prediction
BATCH_SIZE = 32
EPOCHS = 50
VALIDATION_SPLIT = 0.2
LEARNING_RATE = 0.001
