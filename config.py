"""
Configuration file for Cryptocurrency Investment Recommendation System
Supports environment variables for secure deployment
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Helper function to get environment variables with type conversion
def get_env(key, default=None, cast=str):
    """Get environment variable with type casting"""
    value = os.getenv(key, default)
    if value is None:
        return None
    if cast == bool:
        return value.lower() in ('true', '1', 'yes', 'on')
    return cast(value)

# Application settings
APP_ENV = get_env('APP_ENV', 'development')
DEBUG = get_env('DEBUG', True, bool)
SECRET_KEY = get_env('SECRET_KEY', 'dev-secret-key-change-in-production')
DASHBOARD_TITLE = "Cryptocurrency Investment Recommendation System"
FRONTEND_PORT = get_env('FRONTEND_PORT', 8050, int)
API_PORT = get_env('API_PORT', 5000, int)
THEME = get_env('THEME', 'light')

# Investment settings
INITIAL_INVESTMENT = get_env('INITIAL_INVESTMENT', 1000, float)  # CHF
MONTHLY_CONTRIBUTION = get_env('MONTHLY_CONTRIBUTION', 500, float)  # CHF

# Cryptocurrency settings
CRYPTO_SYMBOLS = {
    'BTC-USD': 'Bitcoin',
    'ETH-USD': 'Ethereum'
}

# Game settings
MIN_CASH_AMOUNT = 100  # CHF minimum starting cash
DEFAULT_SIMULATION_LENGTHS = [30, 60, 90]  # days
CUSTOM_SIMULATION_MIN_DAYS = 7
CUSTOM_SIMULATION_MAX_DAYS = 365

# Game states
class GameState:
    SETUP = 'setup'
    PLAYING = 'playing'
    COMPLETED = 'completed'

# Simulation date limits
SIMULATION_EARLIEST_START = '2020-01-01'  # Earliest allowed start date
SIMULATION_BUFFER_DAYS = 30  # Days before today that simulation must end

# Trading settings
TRANSACTION_FEE_PERCENTAGE = 0.1  # 0.1% per transaction
MIN_TRADE_AMOUNT = 10  # CHF minimum trade amount

# Prediction settings
PREDICTION_HORIZONS = [1, 7, 30]  # days

# Directory settings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIRECTORY = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIRECTORY = os.path.join(DATA_DIRECTORY, 'raw')
PROCESSED_DATA_DIRECTORY = os.path.join(DATA_DIRECTORY, 'processed')
SENTIMENT_DATA_DIRECTORY = os.path.join(DATA_DIRECTORY, 'sentiment')
RECOMMENDATIONS_DIRECTORY = os.path.join(DATA_DIRECTORY, 'recommendations')
MODEL_DIRECTORY = os.path.join(BASE_DIR, 'models')
REPORT_DIRECTORY = os.path.join(BASE_DIR, 'reports')
LOG_DIRECTORY = os.path.join(BASE_DIR, 'logs')

# File settings
PORTFOLIO_FILE = os.path.join(DATA_DIRECTORY, 'portfolio.json')
TRANSACTION_HISTORY_FILE = os.path.join(DATA_DIRECTORY, 'transactions.json')

# Database settings (for future use)
DATABASE_URL = get_env('DATABASE_URL', f'sqlite:///{os.path.join(BASE_DIR, "crypto_investment.db")}')

# Create directories if they don't exist
for directory in [DATA_DIRECTORY, RAW_DATA_DIRECTORY, PROCESSED_DATA_DIRECTORY, 
                 SENTIMENT_DATA_DIRECTORY, RECOMMENDATIONS_DIRECTORY, 
                 MODEL_DIRECTORY, REPORT_DIRECTORY, LOG_DIRECTORY]:
    os.makedirs(directory, exist_ok=True)

# Neural network settings
SEQUENCE_LENGTH = get_env('SEQUENCE_LENGTH', 60, int)  # days of historical data
BATCH_SIZE = get_env('BATCH_SIZE', 32, int)
EPOCHS = get_env('EPOCHS', 50, int)
VALIDATION_SPLIT = get_env('VALIDATION_SPLIT', 0.2, float)
TRAIN_TEST_SPLIT = get_env('TRAIN_TEST_SPLIT', 0.8, float)  # 80% train, 20% test
LEARNING_RATE = get_env('LEARNING_RATE', 0.001, float)

# Logging settings
LOG_LEVEL = get_env('LOG_LEVEL', 'INFO')
LOG_RETENTION_DAYS = get_env('LOG_RETENTION_DAYS', 30, int)

# API Keys (for future use)
TWITTER_API_KEY = get_env('TWITTER_API_KEY', '')
TWITTER_API_SECRET = get_env('TWITTER_API_SECRET', '')
TWITTER_ACCESS_TOKEN = get_env('TWITTER_ACCESS_TOKEN', '')
TWITTER_ACCESS_SECRET = get_env('TWITTER_ACCESS_SECRET', '')

# Data settings
DATA_UPDATE_INTERVAL = get_env('DATA_UPDATE_INTERVAL', 3600, int)  # seconds
MAX_HISTORICAL_DAYS = get_env('MAX_HISTORICAL_DAYS', 1825, int)  # 5 years

# Feature flags
ENABLE_REAL_SENTIMENT = get_env('ENABLE_REAL_SENTIMENT', False, bool)
ENABLE_LIVE_TRADING = get_env('ENABLE_LIVE_TRADING', False, bool)
ENABLE_EMAIL_REPORTS = get_env('ENABLE_EMAIL_REPORTS', False, bool)

# Email settings (if email reports are enabled)
SMTP_HOST = get_env('SMTP_HOST', 'smtp.gmail.com')
SMTP_PORT = get_env('SMTP_PORT', 587, int)
SMTP_USERNAME = get_env('SMTP_USERNAME', '')
SMTP_PASSWORD = get_env('SMTP_PASSWORD', '')
EMAIL_FROM = get_env('EMAIL_FROM', 'noreply@crypto-investment.local')
EMAIL_TO = get_env('EMAIL_TO', '')  # Comma-separated list

# Display configuration on startup (only in development)
if DEBUG and APP_ENV == 'development':
    print("="*50)
    print("Configuration Loaded:")
    print(f"Environment: {APP_ENV}")
    print(f"API Port: {API_PORT}")
    print(f"Frontend Port: {FRONTEND_PORT}")
    print(f"Database: {DATABASE_URL}")
    print(f"Real Sentiment: {ENABLE_REAL_SENTIMENT}")
    print(f"Live Trading: {ENABLE_LIVE_TRADING}")
    print("="*50)