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

# Import social sentiment modules
from backend.social_sentiment.reddit.reddit_collector import RedditCollector
from backend.social_sentiment.market_data.market_collector import MarketDataCollector
from backend.social_sentiment.sentiment_analyzer import CryptoSentimentAnalyzer
from backend.social_sentiment.cache_manager import SentimentCacheManager

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

# Initialize social sentiment components
reddit_collector = RedditCollector()
market_collector = MarketDataCollector()
sentiment_analyzer = CryptoSentimentAnalyzer()
cache_manager = SentimentCacheManager()

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

# Social Sentiment endpoints
@app.route('/api/sentiment/social/reddit/current', methods=['GET'])
def get_current_reddit_sentiment():
    """Get current Reddit sentiment data"""
    try:
        coin = request.args.get('coin', 'BTC').upper()
        
        # Check cache first
        today = datetime.now().strftime('%Y-%m-%d')
        cached_data = cache_manager.load_daily_data('reddit', today)
        
        if cached_data:
            return jsonify({
                'status': 'success',
                'data': cached_data,
                'source': 'cache'
            })
        
        # Collect fresh data
        reddit_data = reddit_collector.collect_daily_data()
        
        # Save to cache
        cache_manager.save_daily_data('reddit', today, reddit_data, 
                                    data_sources=['reddit'])
        
        return jsonify({
            'status': 'success',
            'data': reddit_data,
            'source': 'fresh'
        })
        
    except Exception as e:
        logger.error(f"Error getting Reddit sentiment: {str(e)}")
        raise APIError("Failed to get Reddit sentiment data")

@app.route('/api/sentiment/social/market/current', methods=['GET'])
def get_current_market_sentiment():
    """Get current market sentiment data"""
    try:
        coins = request.args.getlist('coins') or ['BTC', 'ETH']
        
        # Check cache first
        today = datetime.now().strftime('%Y-%m-%d')
        cached_data = cache_manager.load_daily_data('market', today)
        
        if cached_data:
            return jsonify({
                'status': 'success',
                'data': cached_data,
                'source': 'cache'
            })
        
        # Collect fresh data
        market_data = market_collector.collect_daily_market_sentiment(coins)
        
        # Save to cache
        cache_manager.save_daily_data('market', today, market_data,
                                    data_sources=['coingecko', 'alternative'])
        
        return jsonify({
            'status': 'success',
            'data': market_data,
            'source': 'fresh'
        })
        
    except Exception as e:
        logger.error(f"Error getting market sentiment: {str(e)}")
        raise APIError("Failed to get market sentiment data")

@app.route('/api/sentiment/social/aggregated', methods=['GET'])
def get_aggregated_sentiment():
    """Get aggregated daily sentiment data"""
    try:
        date = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
        coin = request.args.get('coin', 'BTC').upper()
        
        # Check cache first
        cached_data = cache_manager.load_daily_data('aggregated', date, coin)
        
        if cached_data:
            return jsonify({
                'status': 'success',
                'data': cached_data,
                'source': 'cache'
            })
        
        # Load Reddit and market data for the date
        reddit_data = cache_manager.load_daily_data('reddit', date)
        market_data = cache_manager.load_daily_data('market', date)
        
        if not reddit_data:
            raise APIError(f"No Reddit data available for {date}")
        
        # Convert to post/comment objects and analyze
        from backend.social_sentiment.schemas import RedditPost, RedditComment
        
        posts = [RedditPost(**post) for post in reddit_data.get('posts', [])]
        comments = [RedditComment(**comment) for comment in reddit_data.get('comments', [])]
        
        # Aggregate sentiment
        aggregated = sentiment_analyzer.aggregate_daily_sentiment(posts, comments, coin, date)
        
        # Add market sentiment if available
        if market_data and market_data.get('market_sentiment'):
            market_sentiments = [ms.get('sentiment_score', 0) for ms in market_data['market_sentiment'] 
                               if ms.get('coin') == coin]
            if market_sentiments:
                aggregated.market_sentiment_avg = sum(market_sentiments) / len(market_sentiments)
                # Recalculate combined sentiment
                reddit_weight = 0.7
                market_weight = 0.3
                aggregated.combined_sentiment = (
                    reddit_weight * aggregated.reddit_sentiment_avg + 
                    market_weight * aggregated.market_sentiment_avg
                )
                aggregated.data_sources.append('market')
        
        # Save aggregated data
        aggregated_dict = aggregated.to_dict()
        cache_manager.save_daily_data('aggregated', date, aggregated_dict, coin,
                                    data_sources=aggregated.data_sources)
        
        return jsonify({
            'status': 'success',
            'data': aggregated_dict,
            'source': 'computed'
        })
        
    except Exception as e:
        logger.error(f"Error getting aggregated sentiment: {str(e)}")
        raise APIError("Failed to get aggregated sentiment data")

@app.route('/api/sentiment/social/history', methods=['GET'])
def get_sentiment_history():
    """Get historical sentiment data"""
    try:
        days = request.args.get('days', 7, type=int)
        coin = request.args.get('coin', 'BTC').upper()
        data_type = request.args.get('type', 'aggregated')  # 'reddit', 'market', 'aggregated'
        
        # Get available dates
        available_dates = cache_manager.get_available_dates(data_type)
        
        # Get last N days
        if days > 0:
            available_dates = available_dates[-days:]
        
        history = []
        for date in available_dates:
            if data_type == 'aggregated':
                data = cache_manager.load_daily_data(data_type, date, coin)
            else:
                data = cache_manager.load_daily_data(data_type, date)
            
            if data:
                history.append({
                    'date': date,
                    'data': data
                })
        
        return jsonify({
            'status': 'success',
            'history': history,
            'count': len(history),
            'type': data_type,
            'coin': coin
        })
        
    except Exception as e:
        logger.error(f"Error getting sentiment history: {str(e)}")
        raise APIError("Failed to get sentiment history")

@app.route('/api/sentiment/social/trends', methods=['GET'])
def get_sentiment_trends():
    """Get sentiment trends and analysis"""
    try:
        coin = request.args.get('coin', 'BTC').upper()
        days = request.args.get('days', 7, type=int)
        
        # Get aggregated sentiment history
        available_dates = cache_manager.get_available_dates('aggregated')
        if days > 0:
            available_dates = available_dates[-days:]
        
        trend_data = []
        for date in available_dates:
            data = cache_manager.load_daily_data('aggregated', date, coin)
            if data:
                trend_data.append({
                    'date': date,
                    'sentiment': data.get('combined_sentiment', 0),
                    'confidence': data.get('confidence_level', 0),
                    'signal_strength': data.get('signal_strength', 'weak'),
                    'reddit_posts': data.get('reddit_post_count', 0),
                    'reddit_sentiment': data.get('reddit_sentiment_avg', 0)
                })
        
        # Calculate trend analysis
        sentiments = [d['sentiment'] for d in trend_data]
        trend_analysis = {}
        
        if len(sentiments) >= 2:
            recent_change = sentiments[-1] - sentiments[-2]
            overall_trend = sentiments[-1] - sentiments[0] if len(sentiments) > 1 else 0
            
            trend_analysis = {
                'recent_change': recent_change,
                'overall_trend': overall_trend,
                'average_sentiment': sum(sentiments) / len(sentiments),
                'volatility': max(sentiments) - min(sentiments),
                'trend_direction': 'bullish' if overall_trend > 0.1 else 'bearish' if overall_trend < -0.1 else 'neutral'
            }
        
        return jsonify({
            'status': 'success',
            'trends': trend_data,
            'analysis': trend_analysis,
            'coin': coin,
            'period_days': days
        })
        
    except Exception as e:
        logger.error(f"Error getting sentiment trends: {str(e)}")
        raise APIError("Failed to get sentiment trends")

@app.route('/api/sentiment/social/signals', methods=['GET'])
def get_sentiment_signals():
    """Get sentiment trading signals"""
    try:
        coin = request.args.get('coin', 'BTC').upper()
        
        # Get recent sentiment data
        today = datetime.now().strftime('%Y-%m-%d')
        current_data = cache_manager.load_daily_data('aggregated', today, coin)
        
        if not current_data:
            raise APIError(f"No current sentiment data available for {coin}")
        
        # Generate trading signals based on sentiment
        signals = []
        current_sentiment = current_data.get('combined_sentiment', 0)
        confidence = current_data.get('confidence_level', 0)
        signal_strength = current_data.get('signal_strength', 'weak')
        
        if current_sentiment > 0.5 and confidence > 0.7:
            signals.append({
                'type': 'bullish',
                'strength': 'strong',
                'message': f'Strong bullish sentiment detected for {coin}',
                'confidence': confidence,
                'recommendation': 'Consider buying'
            })
        elif current_sentiment > 0.2 and confidence > 0.6:
            signals.append({
                'type': 'bullish',
                'strength': 'moderate',
                'message': f'Moderate bullish sentiment for {coin}',
                'confidence': confidence,
                'recommendation': 'Monitor for buying opportunity'
            })
        elif current_sentiment < -0.5 and confidence > 0.7:
            signals.append({
                'type': 'bearish',
                'strength': 'strong',
                'message': f'Strong bearish sentiment detected for {coin}',
                'confidence': confidence,
                'recommendation': 'Consider selling or avoid buying'
            })
        elif current_sentiment < -0.2 and confidence > 0.6:
            signals.append({
                'type': 'bearish',
                'strength': 'moderate',
                'message': f'Moderate bearish sentiment for {coin}',
                'confidence': confidence,
                'recommendation': 'Exercise caution'
            })
        else:
            signals.append({
                'type': 'neutral',
                'strength': signal_strength,
                'message': f'Neutral sentiment for {coin}',
                'confidence': confidence,
                'recommendation': 'No clear signal'
            })
        
        return jsonify({
            'status': 'success',
            'signals': signals,
            'coin': coin,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting sentiment signals: {str(e)}")
        raise APIError("Failed to get sentiment signals")

@app.route('/api/sentiment/social/correlation', methods=['GET'])
def get_sentiment_correlation():
    """Get sentiment vs price correlation analysis"""
    try:
        coin = request.args.get('coin', 'BTC').upper()
        days = request.args.get('days', 30, type=int)
        
        # Get sentiment history
        available_dates = cache_manager.get_available_dates('aggregated')
        if days > 0:
            available_dates = available_dates[-days:]
        
        sentiment_data = []
        for date in available_dates:
            data = cache_manager.load_daily_data('aggregated', date, coin)
            if data:
                sentiment_data.append({
                    'date': date,
                    'sentiment': data.get('combined_sentiment', 0)
                })
        
        # Load price data for correlation
        price_file = os.path.join(config.RAW_DATA_DIRECTORY, f'{coin}_USD.csv')
        correlation_data = []
        
        if os.path.exists(price_file):
            import pandas as pd
            price_df = pd.read_csv(price_file)
            price_df['date'] = pd.to_datetime(price_df['date']).dt.strftime('%Y-%m-%d')
            
            # Match sentiment data with price data
            for sentiment_point in sentiment_data:
                price_row = price_df[price_df['date'] == sentiment_point['date']]
                if not price_row.empty:
                    correlation_data.append({
                        'date': sentiment_point['date'],
                        'sentiment': sentiment_point['sentiment'],
                        'price': float(price_row.iloc[0]['close'])
                    })
        
        # Calculate correlation if we have enough data
        correlation_coefficient = None
        if len(correlation_data) >= 5:
            import numpy as np
            sentiments = [d['sentiment'] for d in correlation_data]
            prices = [d['price'] for d in correlation_data]
            correlation_coefficient = np.corrcoef(sentiments, prices)[0, 1]
        
        return jsonify({
            'status': 'success',
            'correlation_data': correlation_data,
            'correlation_coefficient': correlation_coefficient,
            'coin': coin,
            'period_days': len(correlation_data)
        })
        
    except Exception as e:
        logger.error(f"Error getting sentiment correlation: {str(e)}")
        raise APIError("Failed to get sentiment correlation")

# Cache management endpoints
@app.route('/api/sentiment/cache/stats', methods=['GET'])
def get_cache_stats():
    """Get cache statistics"""
    try:
        stats = cache_manager.get_cache_stats()
        return jsonify({
            'status': 'success',
            'cache_stats': stats
        })
    except Exception as e:
        logger.error(f"Error getting cache stats: {str(e)}")
        raise APIError("Failed to get cache statistics")

@app.route('/api/sentiment/cache/cleanup', methods=['POST'])
def cleanup_cache():
    """Clean up old cache data"""
    try:
        retention_days = request.json.get('retention_days', 365) if request.json else 365
        
        # Compress old data
        compressed_count = cache_manager.compress_old_data()
        
        # Clean up very old data
        removed_count = cache_manager.cleanup_old_data(retention_days)
        
        return jsonify({
            'status': 'success',
            'compressed_files': compressed_count,
            'removed_files': removed_count
        })
    except Exception as e:
        logger.error(f"Error cleaning up cache: {str(e)}")
        raise APIError("Failed to clean up cache")

# Legacy sentiment endpoints (keep for backward compatibility)
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