"""
Simple data collector for the crypto trading game
"""

import os
import pandas as pd
import json
from datetime import datetime, timedelta
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class DataCollector:
    """Simple data collector for game functionality"""
    
    def __init__(self):
        self.data_dir = config.RAW_DATA_DIRECTORY
        self.recommendations_dir = config.RECOMMENDATIONS_DIRECTORY
    
    def load_historical_data(self, symbol):
        """Load historical price data for a cryptocurrency"""
        try:
            filename = f"{symbol.replace('-USD', '')}_USD.csv"
            filepath = os.path.join(self.data_dir, filename)
            
            if not os.path.exists(filepath):
                # Return empty DataFrame if file doesn't exist
                return pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
            
            df = pd.read_csv(filepath)
            
            # Ensure Date column is string format
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
            
            return df
            
        except Exception as e:
            print(f"Error loading historical data for {symbol}: {e}")
            return pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    
    def load_recommendations(self, symbol):
        """Load AI recommendations for a cryptocurrency"""
        try:
            filename = f"{symbol.replace('-USD', '')}_recommendations.json"
            filepath = os.path.join(self.recommendations_dir, filename)
            
            if not os.path.exists(filepath):
                return []
            
            with open(filepath, 'r') as f:
                recommendations = json.load(f)
            
            return recommendations
            
        except Exception as e:
            print(f"Error loading recommendations for {symbol}: {e}")
            return []
    
    def get_price_for_date(self, symbol, date):
        """Get price for a specific date"""
        df = self.load_historical_data(symbol)
        
        if df.empty:
            return 0
        
        # Find price for the specific date
        price_row = df[df['Date'] == date]
        
        if price_row.empty:
            # If exact date not found, find the closest earlier date
            df['Date'] = pd.to_datetime(df['Date'])
            target_date = pd.to_datetime(date)
            
            earlier_dates = df[df['Date'] <= target_date]
            if not earlier_dates.empty:
                latest_earlier = earlier_dates.loc[earlier_dates['Date'].idxmax()]
                return float(latest_earlier['Close'])
            
            return 0
        
        return float(price_row.iloc[0]['Close'])