"""
Data preprocessing module for cryptocurrency data
"""

import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import datetime, timedelta

# Import configuration
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def load_data(symbol):
    """Load raw cryptocurrency data from CSV file"""
    filename = os.path.join(config.DATA_DIRECTORY, 'raw', f"{symbol.replace('-', '_')}.csv")
    if os.path.exists(filename):
        df = pd.read_csv(filename, index_col='date', parse_dates=True)
        print(f"Loaded {len(df)} records for {symbol}")
        return df
    else:
        print(f"File not found: {filename}")
        return None

def calculate_technical_indicators(df):
    """Calculate technical indicators for the given DataFrame"""
    # Simple Moving Averages
    df['SMA_7'] = df['close'].rolling(window=7).mean()
    df['SMA_30'] = df['close'].rolling(window=30).mean()
    
    # Exponential Moving Averages
    df['EMA_7'] = df['close'].ewm(span=7, adjust=False).mean()
    df['EMA_30'] = df['close'].ewm(span=30, adjust=False).mean()
    
    # Relative Strength Index (RSI)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Moving Average Convergence Divergence (MACD)
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['BB_middle'] = df['close'].rolling(window=20).mean()
    df['BB_std'] = df['close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (df['BB_std'] * 2)
    df['BB_lower'] = df['BB_middle'] - (df['BB_std'] * 2)
    
    # Percentage change
    df['pct_change_1d'] = df['close'].pct_change(periods=1)
    df['pct_change_7d'] = df['close'].pct_change(periods=7)
    df['pct_change_30d'] = df['close'].pct_change(periods=30)
    
    # Volatility (standard deviation of percentage changes)
    df['volatility_7d'] = df['pct_change_1d'].rolling(window=7).std()
    df['volatility_30d'] = df['pct_change_1d'].rolling(window=30).std()
    
    # Drop NaN values resulting from calculations
    df = df.dropna()
    
    return df

def normalize_data(df):
    """Normalize data using min-max scaling"""
    # Create a copy of the DataFrame to avoid modifying the original
    df_normalized = df.copy()
    
    # Columns to normalize
    columns_to_normalize = ['open', 'high', 'low', 'close', 'volume', 
                           'SMA_7', 'SMA_30', 'EMA_7', 'EMA_30', 
                           'RSI', 'MACD', 'BB_middle', 'BB_upper', 'BB_lower']
    
    # Apply min-max scaling to each column
    for column in columns_to_normalize:
        if column in df_normalized.columns:
            min_val = df_normalized[column].min()
            max_val = df_normalized[column].max()
            df_normalized[f"{column}_norm"] = (df_normalized[column] - min_val) / (max_val - min_val)
    
    return df_normalized

def save_processed_data(df, symbol):
    """Save processed DataFrame to CSV file"""
    if df is not None and not df.empty:
        filename = os.path.join(config.DATA_DIRECTORY, 'processed', f"{symbol.replace('-', '_')}_processed.csv")
        df.to_csv(filename)
        print(f"Saved processed data to {filename}")
        return filename
    return None

def analyze_correlations(dataframes):
    """Analyze correlations between different cryptocurrencies"""
    # Extract close prices from each DataFrame
    close_prices = {}
    for symbol, df in dataframes.items():
        if df is not None and not df.empty:
            close_prices[symbol] = df['close']
    
    # Create a DataFrame with close prices
    correlation_df = pd.DataFrame(close_prices)
    
    # Calculate correlation matrix
    correlation_matrix = correlation_df.corr()
    
    # Save correlation matrix
    filename = os.path.join(config.DATA_DIRECTORY, 'correlation_matrix.csv')
    correlation_matrix.to_csv(filename)
    print(f"Saved correlation matrix to {filename}")
    
    return correlation_matrix

def create_daily_update_script():
    """Create shell script for daily updates"""
    # Use a relative path for the script
    script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'daily_update.sh')
    
    script_content = """#!/bin/bash
# Daily update script for cryptocurrency investment recommendation system

# Navigate to project directory
cd "$(dirname "$0")"

# Activate virtual environment
source venv/bin/activate

# Run data collection and processing
python backend/data_collector.py
python backend/data_processor.py
python backend/mock_sentiment_generator.py

# Train models
python backend/neural_network_model.py

# Generate daily report
python backend/portfolio_tracker.py

echo "Daily update completed at $(date)"
"""
    
    # Create script file
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make script executable
    os.chmod(script_path, 0o755)
    
    print(f"Created daily update script: {script_path}")


def main():
    """Main function to process data for all cryptocurrencies"""
    processed_dataframes = {}
    
    for symbol in config.CRYPTO_SYMBOLS:
        # Load raw data
        df = load_data(symbol)
        
        if df is not None:
            # Calculate technical indicators
            df_with_indicators = calculate_technical_indicators(df)
            
            # Normalize data
            df_normalized = normalize_data(df_with_indicators)
            
            # Save processed data
            save_processed_data(df_normalized, symbol)
            
            processed_dataframes[symbol] = df_normalized
    
    # Analyze correlations between cryptocurrencies
    correlation_matrix = analyze_correlations(processed_dataframes)
    
    # Create daily update script
    create_daily_update_script()
    
    return processed_dataframes, correlation_matrix

if __name__ == "__main__":
    main()
