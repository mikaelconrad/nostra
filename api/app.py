"""
REST API for Cryptocurrency Investment Recommendation System
"""

import os
import sys
from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.logger import setup_logger
from utils.error_handlers import APIError, ValidationError
from utils.validators import (
    CryptoValidator, FinancialValidator, DateValidator,
    validate_transaction_request, validate_data_request
)

# Import backend modules
from backend.data_collector import fetch_crypto_data, update_crypto_data
from backend.sentiment_analyzer import analyze_crypto_sentiment

# Use database-backed portfolio tracker if enabled
if config.get_env('USE_DATABASE', False, bool):
    from backend.portfolio_tracker_db import PortfolioTrackerDB as PortfolioTracker
else:
    from backend.portfolio_tracker import PortfolioTracker

# Set up logger
logger = setup_logger(__name__)

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Initialize portfolio tracker
portfolio_tracker = PortfolioTracker()

# Error handlers
@app.errorhandler(APIError)
def handle_api_error(error):
    logger.error(f"API Error: {str(error)}")
    response = jsonify({'error': str(error)})
    response.status_code = 400
    return response

@app.errorhandler(ValidationError)
def handle_validation_error(error):
    logger.error(f"Validation Error: {str(error)}")
    response = jsonify({'error': str(error)})
    response.status_code = 422
    return response

@app.errorhandler(Exception)
def handle_generic_error(error):
    logger.error(f"Unexpected error: {str(error)}", exc_info=True)
    response = jsonify({'error': 'Internal server error'})
    response.status_code = 500
    return response

# Health check endpoint
@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

# Cryptocurrency data endpoints
@app.route('/api/crypto/<symbol>/data', methods=['GET'])
def get_crypto_data(symbol):
    """Get cryptocurrency data"""
    try:
        # Validate symbol
        symbol = CryptoValidator.validate_symbol(symbol)
        
        # Validate query parameters
        params = {'symbol': symbol}
        if 'days' in request.args:
            params['days'] = request.args.get('days', type=int)
        
        validated_params = validate_data_request(params)
        days = validated_params.get('days', 30)
        
        # Load data from CSV
        filepath = os.path.join(config.RAW_DATA_DIRECTORY, f"{symbol}_USD.csv")
        if not os.path.exists(filepath):
            raise APIError(f"No data available for {symbol}")
        
        import pandas as pd
        df = pd.read_csv(filepath)
        
        # Limit to requested days
        if days > 0:
            df = df.tail(days)
        
        # Convert to JSON
        data = df.to_dict('records')
        
        return jsonify({
            'symbol': symbol,
            'data': data,
            'count': len(data)
        })
        
    except ValidationError:
        raise
    except Exception as e:
        logger.error(f"Error getting data for {symbol}: {str(e)}")
        raise APIError(f"Failed to get data for {symbol}")

@app.route('/api/crypto/<symbol>/update', methods=['POST'])
def update_crypto(symbol):
    """Update cryptocurrency data"""
    try:
        # Validate symbol
        symbol = symbol.upper()
        validate_input(symbol, is_valid_symbol, f"Invalid cryptocurrency symbol: {symbol}")
        
        # Get days parameter
        days = request.json.get('days', 7) if request.json else 7
        
        # Update data
        update_crypto_data(f"{symbol}-USD", days)
        
        return jsonify({
            'status': 'success',
            'message': f'Updated {symbol} data for last {days} days'
        })
        
    except Exception as e:
        logger.error(f"Error updating {symbol}: {str(e)}")
        raise APIError(f"Failed to update {symbol}")

# Portfolio endpoints
@app.route('/api/portfolio', methods=['GET'])
def get_portfolio():
    """Get current portfolio"""
    try:
        portfolio = portfolio_tracker.portfolio
        
        # Get current prices
        current_prices = {}
        for symbol in ['BTC', 'ETH']:
            price_file = os.path.join(config.RAW_DATA_DIRECTORY, f'{symbol}_USD.csv')
            if os.path.exists(price_file):
                import pandas as pd
                df = pd.read_csv(price_file)
                if not df.empty:
                    current_prices[symbol] = df.iloc[-1]['close']
        
        # Calculate metrics
        metrics = portfolio_tracker.calculate_performance_metrics(current_prices)
        
        # Get holdings with values
        holdings = portfolio_tracker.get_holdings_by_symbol()
        for symbol in holdings:
            if symbol != 'cash' and symbol in current_prices:
                holdings[symbol]['value'] = holdings[symbol]['amount'] * current_prices[symbol]
        
        return jsonify({
            'portfolio': portfolio,
            'holdings': holdings,
            'metrics': metrics,
            'current_prices': current_prices
        })
        
    except Exception as e:
        logger.error(f"Error getting portfolio: {str(e)}")
        raise APIError("Failed to get portfolio")

@app.route('/api/portfolio/buy', methods=['POST'])
def buy_crypto():
    """Buy cryptocurrency"""
    try:
        # Validate request data
        validated_data = validate_transaction_request(request.json or {})
        
        # Execute buy
        success, message = portfolio_tracker.buy_crypto(
            validated_data['symbol'], 
            validated_data['amount'], 
            validated_data['price']
        )
        
        if success:
            return jsonify({
                'status': 'success',
                'message': message,
                'portfolio': portfolio_tracker.portfolio
            })
        else:
            raise APIError(message)
            
    except ValidationError:
        raise  # Re-raise validation errors to be handled by error handler
    except Exception as e:
        logger.error(f"Error buying crypto: {str(e)}")
        raise APIError(f"Failed to execute buy order")

@app.route('/api/portfolio/sell', methods=['POST'])
def sell_crypto():
    """Sell cryptocurrency"""
    try:
        # Validate request data
        validated_data = validate_transaction_request(request.json or {})
        
        # Execute sell
        success, message = portfolio_tracker.sell_crypto(
            validated_data['symbol'], 
            validated_data['amount'], 
            validated_data['price']
        )
        
        if success:
            return jsonify({
                'status': 'success',
                'message': message,
                'portfolio': portfolio_tracker.portfolio
            })
        else:
            raise APIError(message)
            
    except ValidationError:
        raise  # Re-raise validation errors to be handled by error handler
    except Exception as e:
        logger.error(f"Error selling crypto: {str(e)}")
        raise APIError(f"Failed to execute sell order")

@app.route('/api/portfolio/transactions', methods=['GET'])
def get_transactions():
    """Get transaction history"""
    try:
        transactions = portfolio_tracker.load_transactions()
        summary = portfolio_tracker.get_transaction_summary()
        
        return jsonify({
            'transactions': transactions,
            'summary': summary
        })
        
    except Exception as e:
        logger.error(f"Error getting transactions: {str(e)}")
        raise APIError("Failed to get transactions")

# Recommendations endpoints
@app.route('/api/recommendations', methods=['GET'])
def get_recommendations():
    """Get investment recommendations"""
    try:
        recommendations = {}
        
        for symbol in ['BTC', 'ETH']:
            rec_file = os.path.join(config.RECOMMENDATIONS_DIRECTORY, f'{symbol}_recommendations.json')
            if os.path.exists(rec_file):
                with open(rec_file, 'r') as f:
                    recommendations[symbol] = json.load(f)
        
        return jsonify({
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}")
        raise APIError("Failed to get recommendations")

# Sentiment endpoints
@app.route('/api/sentiment/<symbol>', methods=['GET'])
def get_sentiment(symbol):
    """Get sentiment analysis for cryptocurrency"""
    try:
        # Validate symbol
        symbol = CryptoValidator.validate_symbol(symbol)
        
        # Load sentiment data
        sentiment_file = os.path.join(config.SENTIMENT_DATA_DIRECTORY, f'{symbol}_daily_sentiment.csv')
        if not os.path.exists(sentiment_file):
            raise APIError(f"No sentiment data available for {symbol}")
        
        import pandas as pd
        df = pd.read_csv(sentiment_file)
        
        # Get last N days
        days = request.args.get('days', 7, type=int)
        if days > 0:
            df = df.tail(days)
        
        # Convert to JSON
        data = df.to_dict('records')
        
        return jsonify({
            'symbol': symbol,
            'sentiment': data,
            'count': len(data)
        })
        
    except Exception as e:
        logger.error(f"Error getting sentiment for {symbol}: {str(e)}")
        raise APIError(f"Failed to get sentiment for {symbol}")

# Report endpoints
@app.route('/api/reports/generate', methods=['POST'])
def generate_report():
    """Generate daily report"""
    try:
        report_path = portfolio_tracker.generate_daily_report()
        
        return jsonify({
            'status': 'success',
            'report_path': report_path,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        raise APIError("Failed to generate report")

# Process data endpoint
@app.route('/api/process', methods=['POST'])
def process_data():
    """Process all cryptocurrency data"""
    try:
        # Run data processing
        from backend.integration import run_data_processing
        run_data_processing()
        
        return jsonify({
            'status': 'success',
            'message': 'Data processing completed',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        raise APIError("Failed to process data")

def run_api():
    """Run the Flask API server"""
    logger.info("Starting Cryptocurrency Investment API...")
    logger.info(f"API running on http://localhost:{config.API_PORT}")
    app.run(
        host='0.0.0.0',
        port=config.API_PORT,
        debug=False  # Set to False in production
    )

if __name__ == '__main__':
    run_api()