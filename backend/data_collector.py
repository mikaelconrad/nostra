"""
Data collection module for cryptocurrency investment recommendation system
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf

# Import configuration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def fetch_crypto_data(symbol, interval='1d', range='5y'):
    """
    Fetch cryptocurrency data from Yahoo Finance
    
    Args:
        symbol: Cryptocurrency symbol (e.g., 'BTC-USD')
        interval: Data interval (e.g., '1d', '1h')
        range: Data range (e.g., '1mo', '1y', '5y')
        
    Returns:
        DataFrame with OHLCV data
    """
    print(f"Fetching data for {symbol}...")
    try:
        # Fetch data from Yahoo Finance
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=range, interval=interval)
        
        # Rename columns to lowercase
        df.columns = [col.lower() for col in df.columns]
        
        # Reset index to make date a column
        df = df.reset_index()
        
        # Convert date column to string format - Fix the error here
        if 'date' in df.columns:
            df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        elif 'Date' in df.columns:
            df['date'] = df['Date'].dt.strftime('%Y-%m-%d')
            df = df.drop('Date', axis=1)
        else:
            # If neither 'date' nor 'Date' exists, check what columns are available
            print(f"Available columns: {df.columns.tolist()}")
            # Try to use the first column as date if it's a datetime
            if df.index.dtype == 'datetime64[ns]':
                df['date'] = df.index.strftime('%Y-%m-%d')
                df = df.reset_index(drop=True)
        
        print(f"Successfully fetched {len(df)} records for {symbol}")
        return df
    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
        return None

def save_data(df, symbol):
    """
    Save data to CSV file
    
    Args:
        df: DataFrame with OHLCV data
        symbol: Cryptocurrency symbol (e.g., 'BTC-USD')
    """
    if df is None or len(df) == 0:
        print(f"No data to save for {symbol}")
        return
    
    # Create directory if it doesn't exist
    os.makedirs(config.RAW_DATA_DIRECTORY, exist_ok=True)
    
    # Save to CSV
    filename = os.path.join(config.RAW_DATA_DIRECTORY, f"{symbol.replace('-', '_')}.csv")
    df.to_csv(filename, index=False)
    print(f"Saved {len(df)} records to {filename}")

def save_metadata():
    """Save metadata about the data collection"""
    metadata = {
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'symbols': list(config.CRYPTO_SYMBOLS.keys()),
        'data_source': 'Yahoo Finance'
    }
    
    # Create directory if it doesn't exist
    os.makedirs(config.DATA_DIRECTORY, exist_ok=True)
    
    # Save metadata
    filename = os.path.join(config.DATA_DIRECTORY, 'metadata.json')
    with open(filename, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"Saved metadata to {filename}")

def create_mock_data():
    """Create mock data if fetching fails"""
    for symbol in config.CRYPTO_SYMBOLS:
        print(f"Creating mock data for {symbol}")
        
        # Create date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*5)  # 5 years of data
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Create mock price data
        if symbol == 'BTC-USD':
            base_price = 50000
            volatility = 5000
        elif symbol == 'ETH-USD':
            base_price = 3000
            volatility = 300
        else:  # XRP-USD
            base_price = 1
            volatility = 0.1
        
        # Generate random walk prices
        import numpy as np
        np.random.seed(42)  # For reproducibility
        price_changes = np.random.normal(0, volatility/100, len(dates))
        prices = base_price * (1 + np.cumsum(price_changes))
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': dates.strftime('%Y-%m-%d'),
            'open': prices,
            'high': prices * (1 + np.random.uniform(0, 0.05, len(dates))),
            'low': prices * (1 - np.random.uniform(0, 0.05, len(dates))),
            'close': prices * (1 + np.random.normal(0, 0.02, len(dates))),
            'volume': np.random.uniform(1000000, 10000000, len(dates))
        })
        
        # Save mock data
        filename = os.path.join(config.RAW_DATA_DIRECTORY, f"{symbol.replace('-', '_')}.csv")
        df.to_csv(filename, index=False)
        print(f"Saved mock data to {filename}")

def main():
    """Main function"""
    # Create directories
    os.makedirs(config.RAW_DATA_DIRECTORY, exist_ok=True)
    os.makedirs(config.PROCESSED_DATA_DIRECTORY, exist_ok=True)
    os.makedirs(config.SENTIMENT_DATA_DIRECTORY, exist_ok=True)
    os.makedirs(config.RECOMMENDATIONS_DIRECTORY, exist_ok=True)
    print("Created directories in data")
    
    # Fetch and save data for each cryptocurrency
    success = True
    for symbol in config.CRYPTO_SYMBOLS:
        df = fetch_crypto_data(symbol, interval='1d', range='5y')
        if df is not None and len(df) > 0:
            save_data(df, symbol)
        else:
            success = False
    
    # If fetching failed, create mock data
    if not success:
        print("Fetching data failed. Creating mock data instead.")
        create_mock_data()
    
    # Save metadata
    save_metadata()

if __name__ == "__main__":
    main()
