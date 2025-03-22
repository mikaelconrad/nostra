"""
Integration script to connect backend and frontend components
"""

import sys
import os
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import subprocess
import time
import threading

# Import configuration
sys.path.append('/home/ubuntu/crypto_investment_app')
import config

def run_data_collection():
    """Run data collection script"""
    print("Running data collection...")
    subprocess.run(["python", "backend/data_collector.py"], 
                  cwd="/home/ubuntu/crypto_investment_app")
    print("Data collection completed")

def run_data_processing():
    """Run data processing script"""
    print("Running data processing...")
    subprocess.run(["python", "backend/data_processor.py"], 
                  cwd="/home/ubuntu/crypto_investment_app")
    print("Data processing completed")

def run_sentiment_analysis():
    """Run sentiment analysis script"""
    print("Running sentiment analysis...")
    subprocess.run(["python", "backend/mock_sentiment_generator.py"], 
                  cwd="/home/ubuntu/crypto_investment_app")
    print("Sentiment analysis completed")

def run_model_training():
    """Run model training script"""
    print("Running model training...")
    # Create model directory if it doesn't exist
    os.makedirs(config.MODEL_DIRECTORY, exist_ok=True)
    
    # For demonstration purposes, we'll create mock recommendation files
    # In a real scenario, this would run the neural network training
    create_mock_recommendations()
    print("Model training completed")

def create_mock_recommendations():
    """Create mock recommendation files for demonstration"""
    # Create recommendations directory
    recommendations_dir = os.path.join(config.DATA_DIRECTORY, 'recommendations')
    os.makedirs(recommendations_dir, exist_ok=True)
    
    # Current prices (from recent data)
    current_prices = {
        'BTC': 68000,
        'ETH': 3500,
        'XRP': 0.5
    }
    
    # Generate recommendations for each cryptocurrency
    for symbol in config.CRYPTO_SYMBOLS:
        base_symbol = symbol.split('-')[0]
        current_price = current_prices.get(base_symbol, 1000)
        
        recommendations = {}
        
        # Generate recommendations for each prediction horizon
        for horizon in config.PREDICTION_HORIZONS:
            # Generate random prediction with bias towards positive returns for demonstration
            random_factor = np.random.normal(1.05, 0.1)
            predicted_price = current_price * random_factor
            expected_return = (predicted_price - current_price) / current_price * 100
            
            # Determine recommendation
            if expected_return > 5:
                action = "BUY"
                strength = "STRONG"
            elif expected_return > 1:
                action = "BUY"
                strength = "MODERATE"
            elif expected_return > -1:
                action = "HOLD"
                strength = "NEUTRAL"
            elif expected_return > -5:
                action = "SELL"
                strength = "MODERATE"
            else:
                action = "SELL"
                strength = "STRONG"
            
            # Add to recommendations
            recommendations[str(horizon)] = {
                'current_price': float(current_price),
                'predicted_price': float(predicted_price),
                'expected_return': float(expected_return),
                'action': action,
                'strength': strength,
                'prediction_date': (datetime.now() + timedelta(days=horizon)).strftime('%Y-%m-%d')
            }
        
        # Save recommendations
        filename = os.path.join(recommendations_dir, f"{base_symbol}_recommendations.json")
        with open(filename, 'w') as f:
            json.dump(recommendations, f, indent=4)
        
        print(f"Created mock recommendations for {base_symbol}")

def create_initial_portfolio():
    """Create initial portfolio file if it doesn't exist"""
    if not os.path.exists(config.PORTFOLIO_FILE):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(config.PORTFOLIO_FILE), exist_ok=True)
        
        # Create default portfolio
        portfolio = {
            'initial_investment': config.INITIAL_INVESTMENT,
            'monthly_contribution': config.MONTHLY_CONTRIBUTION,
            'total_invested': config.INITIAL_INVESTMENT,
            'current_value': config.INITIAL_INVESTMENT,
            'holdings': {
                'BTC-USD': {'amount': 0, 'value': 0},
                'ETH-USD': {'amount': 0, 'value': 0},
                'XRP-USD': {'amount': 0, 'value': 0},
                'cash': config.INITIAL_INVESTMENT
            },
            'transactions': []
        }
        
        # Save portfolio
        with open(config.PORTFOLIO_FILE, 'w') as f:
            json.dump(portfolio, f, indent=4)
        
        print(f"Created initial portfolio file at {config.PORTFOLIO_FILE}")

def run_frontend():
    """Run frontend application"""
    print("Starting frontend application...")
    subprocess.Popen(["python", "frontend/app.py"], 
                    cwd="/home/ubuntu/crypto_investment_app")
    print(f"Frontend application started at http://localhost:{config.FRONTEND_PORT}")

def setup_daily_updates():
    """Set up daily update script"""
    script_content = """#!/bin/bash
# Daily update script for cryptocurrency investment recommendation system

cd /home/ubuntu/crypto_investment_app
source venv/bin/activate

# Run data collection and processing
python backend/data_collector.py
python backend/data_processor.py

# Run sentiment analysis
python backend/mock_sentiment_generator.py

# Generate new recommendations
python backend/integration.py --recommendations-only

echo "Daily update completed at $(date)"
"""
    
    script_path = os.path.join('/home/ubuntu/crypto_investment_app', 'daily_update.sh')
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make the script executable
    os.chmod(script_path, 0o755)
    
    print(f"Created daily update script at {script_path}")
    
    # Create a crontab entry instruction
    crontab_instruction = """
# To set up automatic daily updates, run the following command:
# (crontab -l 2>/dev/null; echo "0 0 * * * /home/ubuntu/crypto_investment_app/daily_update.sh >> /home/ubuntu/crypto_investment_app/daily_update.log 2>&1") | crontab -
# This will run the update script every day at midnight
"""
    
    instruction_path = os.path.join('/home/ubuntu/crypto_investment_app', 'crontab_setup.txt')
    with open(instruction_path, 'w') as f:
        f.write(crontab_instruction)
    
    print(f"Created crontab setup instructions at {instruction_path}")

def main():
    """Main integration function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Cryptocurrency Investment Recommendation System Integration')
    parser.add_argument('--recommendations-only', action='store_true', help='Only generate recommendations')
    args = parser.parse_args()
    
    if args.recommendations_only:
        # Only generate recommendations
        create_mock_recommendations()
    else:
        # Run full integration
        print("Starting integration of backend and frontend components...")
        
        # Run data collection
        run_data_collection()
        
        # Run data processing
        run_data_processing()
        
        # Run sentiment analysis
        run_sentiment_analysis()
        
        # Run model training and generate recommendations
        run_model_training()
        
        # Create initial portfolio
        create_initial_portfolio()
        
        # Set up daily updates
        setup_daily_updates()
        
        # Run frontend application
        run_frontend()
        
        print("Integration completed successfully")

if __name__ == "__main__":
    main()
