#!/usr/bin/env python3
"""
Test script for Cryptocurrency Investment Recommendation System
"""

import sys
import os
import unittest
import json
import pandas as pd
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

class TestCryptoInvestmentSystem(unittest.TestCase):
    """Test cases for Cryptocurrency Investment Recommendation System"""
    
    def test_data_collection(self):
        """Test data collection functionality"""
        # Import data collector
        sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend'))
        from data_collector import fetch_crypto_data
        
        # Test data collection for BTC
        df = fetch_crypto_data('BTC-USD', interval='1d', range='1mo')
        
        # Check if data was fetched successfully
        self.assertIsNotNone(df)
        self.assertGreater(len(df), 0)
        
        # Check if required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            self.assertIn(col, df.columns)
    
    def test_data_processing(self):
        """Test data processing functionality"""
        # Check if processed data files exist
        for symbol in config.CRYPTO_SYMBOLS:
            filename = os.path.join(config.DATA_DIRECTORY, 'processed', f"{symbol.replace('-', '_')}_processed.csv")
            self.assertTrue(os.path.exists(filename), f"Processed data file not found: {filename}")
            
            # Load data and check for technical indicators
            df = pd.read_csv(filename)
            technical_indicators = ['SMA_7', 'SMA_30', 'EMA_7', 'EMA_30', 'RSI', 'MACD']
            for indicator in technical_indicators:
                self.assertIn(indicator, df.columns, f"Technical indicator {indicator} not found in processed data")
    
    def test_sentiment_analysis(self):
        """Test sentiment analysis functionality"""
        # Check if sentiment data files exist
        for symbol in config.CRYPTO_SYMBOLS:
            base_symbol = symbol.split('-')[0]
            filename = os.path.join(config.SENTIMENT_DATA_DIRECTORY, f"{base_symbol}_daily_sentiment.csv")
            self.assertTrue(os.path.exists(filename), f"Sentiment data file not found: {filename}")
            
            # Load data and check for sentiment scores
            df = pd.read_csv(filename)
            sentiment_columns = ['vader_compound', 'textblob_polarity']
            for col in sentiment_columns:
                self.assertIn(col, df.columns, f"Sentiment column {col} not found in sentiment data")
    
    def test_recommendations(self):
        """Test recommendation generation"""
        # Check if recommendation files exist
        recommendations_dir = os.path.join(config.DATA_DIRECTORY, 'recommendations')
        self.assertTrue(os.path.exists(recommendations_dir), "Recommendations directory not found")
        
        for symbol in config.CRYPTO_SYMBOLS:
            base_symbol = symbol.split('-')[0]
            filename = os.path.join(recommendations_dir, f"{base_symbol}_recommendations.json")
            self.assertTrue(os.path.exists(filename), f"Recommendations file not found: {filename}")
            
            # Load recommendations and check structure
            with open(filename, 'r') as f:
                recommendations = json.load(f)
            
            # Check if recommendations exist for all prediction horizons
            for horizon in config.PREDICTION_HORIZONS:
                self.assertIn(str(horizon), recommendations, f"Recommendations for {horizon}-day horizon not found")
                
                # Check recommendation structure
                rec = recommendations[str(horizon)]
                self.assertIn('action', rec, "Action not found in recommendation")
                self.assertIn('strength', rec, "Strength not found in recommendation")
                self.assertIn('current_price', rec, "Current price not found in recommendation")
                self.assertIn('predicted_price', rec, "Predicted price not found in recommendation")
                self.assertIn('expected_return', rec, "Expected return not found in recommendation")
    
    def test_portfolio_tracking(self):
        """Test portfolio tracking functionality"""
        # Check if portfolio file exists
        self.assertTrue(os.path.exists(config.PORTFOLIO_FILE), "Portfolio file not found")
        
        # Load portfolio and check structure
        with open(config.PORTFOLIO_FILE, 'r') as f:
            portfolio = json.load(f)
        
        # Check portfolio structure
        self.assertIn('initial_investment', portfolio, "Initial investment not found in portfolio")
        self.assertIn('monthly_contribution', portfolio, "Monthly contribution not found in portfolio")
        self.assertIn('holdings', portfolio, "Holdings not found in portfolio")
        self.assertIn('transactions', portfolio, "Transactions not found in portfolio")
        
        # Check if holdings include all cryptocurrencies
        for symbol in config.CRYPTO_SYMBOLS:
            self.assertIn(symbol, portfolio['holdings'], f"Cryptocurrency {symbol} not found in portfolio holdings")
    
    def test_reporting(self):
        """Test reporting functionality"""
        # Check if report directory exists
        self.assertTrue(os.path.exists(config.REPORT_DIRECTORY), "Report directory not found")
        
        # Import portfolio tracker
        sys.path.append('/home/ubuntu/crypto_investment_app/backend')
        from portfolio_tracker import PortfolioTracker
        
        # Create portfolio tracker and generate report
        tracker = PortfolioTracker()
        report_file = tracker.generate_daily_report()
        
        # Check if report file was created
        self.assertTrue(os.path.exists(report_file), "Report file not created")
        
        # Check report file size
        self.assertGreater(os.path.getsize(report_file), 0, "Report file is empty")

def run_tests():
    """Run all tests"""
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

if __name__ == "__main__":
    run_tests()
