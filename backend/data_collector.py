"""
Data collection module for cryptocurrency investment recommendation system
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf

# Import configuration and utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.logger import setup_logger, log_exception
from utils.error_handlers import DataCollectionError, handle_error, safe_execute

# Set up logger
logger = setup_logger(__name__)

@log_exception(logger)
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
    logger.info(f"Fetching data for {symbol} with interval={interval}, range={range}")
    
    try:
        # Fetch data from Yahoo Finance
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=range, interval=interval)
        
        if df.empty:
            handle_error(DataCollectionError, f"No data returned for {symbol}")
        
        # Rename columns to lowercase
        df.columns = [col.lower() for col in df.columns]
        
        # Reset index to make date a column
        df = df.reset_index()
        
        # Convert date column to string format
        if 'date' in df.columns:
            df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        elif 'Date' in df.columns:
            df['date'] = df['Date'].dt.strftime('%Y-%m-%d')
            df = df.drop('Date', axis=1)
        else:
            # If neither 'date' nor 'Date' exists, check what columns are available
            logger.warning(f"Date column not found. Available columns: {df.columns.tolist()}")
            # Try to use the first column as date if it's a datetime
            if df.index.dtype == 'datetime64[ns]':
                df['date'] = df.index.strftime('%Y-%m-%d')
        
        logger.info(f"Successfully fetched {len(df)} records for {symbol}")
        return df
        
    except Exception as e:
        handle_error(DataCollectionError, f"Failed to fetch data for {symbol}", e)

@log_exception(logger)
def save_crypto_data(df, symbol):
    """
    Save cryptocurrency data to CSV file
    
    Args:
        df: DataFrame with cryptocurrency data
        symbol: Cryptocurrency symbol
    """
    try:
        # Clean symbol for filename (remove -USD suffix)
        clean_symbol = symbol.split('-')[0]
        
        # Create filename
        filename = f"{clean_symbol}_USD.csv"
        filepath = os.path.join(config.RAW_DATA_DIRECTORY, filename)
        
        # Save to CSV
        df.to_csv(filepath, index=False)
        logger.info(f"Data saved to {filepath}")
        
        # Save metadata
        save_metadata(symbol, len(df), df['date'].min(), df['date'].max())
        
    except Exception as e:
        handle_error(DataCollectionError, f"Failed to save data for {symbol}", e)

@log_exception(logger)
def save_metadata(symbol, record_count, start_date, end_date):
    """
    Save metadata about collected data
    
    Args:
        symbol: Cryptocurrency symbol
        record_count: Number of records
        start_date: Start date of data
        end_date: End date of data
    """
    try:
        metadata_file = os.path.join(config.DATA_DIRECTORY, 'metadata.json')
        
        # Load existing metadata
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        # Update metadata
        metadata[symbol] = {
            'last_updated': datetime.now().isoformat(),
            'record_count': record_count,
            'start_date': start_date,
            'end_date': end_date
        }
        
        # Save metadata
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        logger.debug(f"Metadata saved for {symbol}")
        
    except Exception as e:
        logger.warning(f"Failed to save metadata for {symbol}: {str(e)}")
        # Don't raise error for metadata as it's not critical

@log_exception(logger)
def collect_all_crypto_data():
    """
    Collect data for all configured cryptocurrencies
    """
    logger.info("Starting data collection for all cryptocurrencies")
    
    successful = []
    failed = []
    
    for symbol, name in config.CRYPTO_SYMBOLS.items():
        try:
            logger.info(f"Processing {name} ({symbol})...")
            
            # Fetch data
            df = fetch_crypto_data(symbol)
            
            # Save data
            save_crypto_data(df, symbol)
            
            successful.append(symbol)
            
        except DataCollectionError as e:
            logger.error(f"Failed to process {symbol}: {str(e)}")
            failed.append(symbol)
        except Exception as e:
            logger.error(f"Unexpected error processing {symbol}: {str(e)}", exc_info=True)
            failed.append(symbol)
    
    # Summary
    logger.info(f"Data collection completed. Successful: {len(successful)}, Failed: {len(failed)}")
    if failed:
        logger.warning(f"Failed symbols: {', '.join(failed)}")
    
    return successful, failed

@log_exception(logger)
def update_crypto_data(symbol, days_back=7):
    """
    Update cryptocurrency data with recent values
    
    Args:
        symbol: Cryptocurrency symbol
        days_back: Number of days to fetch
    """
    logger.info(f"Updating data for {symbol} (last {days_back} days)")
    
    try:
        # Fetch recent data
        df = fetch_crypto_data(symbol, range=f'{days_back}d')
        
        # Load existing data
        clean_symbol = symbol.split('-')[0]
        filepath = os.path.join(config.RAW_DATA_DIRECTORY, f"{clean_symbol}_USD.csv")
        
        if os.path.exists(filepath):
            existing_df = pd.read_csv(filepath)
            
            # Merge with new data (removing duplicates)
            combined_df = pd.concat([existing_df, df])
            combined_df = combined_df.drop_duplicates(subset=['date'], keep='last')
            combined_df = combined_df.sort_values('date')
            
            # Save updated data
            combined_df.to_csv(filepath, index=False)
            logger.info(f"Updated {symbol} data with {len(df)} new records")
        else:
            # No existing data, save as new
            save_crypto_data(df, symbol)
            
    except Exception as e:
        handle_error(DataCollectionError, f"Failed to update data for {symbol}", e)

def main():
    """Main function to collect cryptocurrency data"""
    logger.info("="*50)
    logger.info("Cryptocurrency Data Collection")
    logger.info("="*50)
    
    # Create directories
    os.makedirs(config.RAW_DATA_DIRECTORY, exist_ok=True)
    
    # Collect data for all cryptocurrencies
    successful, failed = collect_all_crypto_data()
    
    # Return status
    return len(failed) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)