"""
Tests for the REST API endpoints
"""

import pytest
import json
import os
from unittest.mock import patch, MagicMock

class TestAPIHealthCheck:
    """Test suite for health check endpoint"""
    
    def test_health_check(self, client):
        """Test the health check endpoint"""
        response = client.get('/api/health')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert 'timestamp' in data
        assert data['version'] == '1.0.0'

class TestAPICryptoData:
    """Test suite for cryptocurrency data endpoints"""
    
    def test_get_crypto_data_valid(self, client, temp_data_dir, sample_price_data):
        """Test getting cryptocurrency data"""
        # Save sample data
        btc_data = sample_price_data['BTC']
        btc_data.to_csv(os.path.join(temp_data_dir, 'raw', 'BTC_USD.csv'), index=False)
        
        response = client.get('/api/crypto/BTC/data?days=10')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['symbol'] == 'BTC'
        assert len(data['data']) == 10
        assert data['count'] == 10
    
    def test_get_crypto_data_invalid_symbol(self, client):
        """Test getting data for invalid symbol"""
        response = client.get('/api/crypto/INVALID/data')
        
        assert response.status_code == 422
        data = json.loads(response.data)
        assert 'error' in data
        assert 'Invalid cryptocurrency symbol' in data['error']
    
    def test_get_crypto_data_no_file(self, client, temp_data_dir):
        """Test getting data when file doesn't exist"""
        response = client.get('/api/crypto/BTC/data')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'No data available' in data['error']
    
    @patch('backend.data_collector.update_crypto_data')
    def test_update_crypto_data(self, mock_update, client):
        """Test updating cryptocurrency data"""
        mock_update.return_value = None
        
        response = client.post('/api/crypto/BTC/update', 
                              json={'days': 7})
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'success'
        mock_update.assert_called_once_with('BTC-USD', 7)

class TestAPIPortfolio:
    """Test suite for portfolio endpoints"""
    
    def test_get_portfolio(self, client, portfolio_tracker, sample_portfolio_data, temp_data_dir):
        """Test getting portfolio data"""
        # Set up portfolio
        portfolio_tracker.portfolio = sample_portfolio_data
        portfolio_tracker.save_portfolio()
        
        # Save some price data
        import pandas as pd
        for symbol in ['BTC', 'ETH', 'XRP']:
            df = pd.DataFrame({
                'date': ['2024-01-01'],
                'close': [50000 if symbol == 'BTC' else 2000 if symbol == 'ETH' else 0.5]
            })
            df.to_csv(os.path.join(temp_data_dir, 'raw', f'{symbol}_USD.csv'), index=False)
        
        response = client.get('/api/portfolio')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'portfolio' in data
        assert 'holdings' in data
        assert 'metrics' in data
        assert 'current_prices' in data
    
    def test_buy_crypto_success(self, client, portfolio_tracker):
        """Test successful cryptocurrency purchase"""
        # Set initial cash
        portfolio_tracker.portfolio['cash'] = 10000.0
        portfolio_tracker.save_portfolio()
        
        response = client.post('/api/portfolio/buy',
                              json={
                                  'symbol': 'BTC',
                                  'amount': '0.1',
                                  'price': '50000'
                              })
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'success'
        assert 'Bought 0.1 BTC' in data['message']
        assert data['portfolio']['cash'] == 5000.0
        assert data['portfolio']['holdings']['BTC'] == 0.1
    
    def test_buy_crypto_invalid_data(self, client):
        """Test buy with invalid data"""
        # Missing required fields
        response = client.post('/api/portfolio/buy', json={})
        assert response.status_code == 422
        
        # Invalid symbol
        response = client.post('/api/portfolio/buy',
                              json={
                                  'symbol': 'INVALID',
                                  'amount': '0.1',
                                  'price': '50000'
                              })
        assert response.status_code == 422
        data = json.loads(response.data)
        assert 'Invalid cryptocurrency symbol' in data['error']
        
        # Negative amount
        response = client.post('/api/portfolio/buy',
                              json={
                                  'symbol': 'BTC',
                                  'amount': '-0.1',
                                  'price': '50000'
                              })
        assert response.status_code == 422
    
    def test_sell_crypto_success(self, client, portfolio_tracker):
        """Test successful cryptocurrency sale"""
        # Set up portfolio with BTC
        portfolio_tracker.portfolio = {
            'cash': 1000.0,
            'holdings': {'BTC': 0.2}
        }
        portfolio_tracker.save_portfolio()
        
        response = client.post('/api/portfolio/sell',
                              json={
                                  'symbol': 'BTC',
                                  'amount': '0.1',
                                  'price': '55000'
                              })
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'success'
        assert 'Sold 0.1 BTC' in data['message']
        assert data['portfolio']['cash'] == 6500.0
        assert data['portfolio']['holdings']['BTC'] == 0.1
    
    def test_sell_crypto_insufficient_holdings(self, client, portfolio_tracker):
        """Test sell with insufficient holdings"""
        # Empty portfolio
        portfolio_tracker.portfolio = {
            'cash': 1000.0,
            'holdings': {}
        }
        portfolio_tracker.save_portfolio()
        
        response = client.post('/api/portfolio/sell',
                              json={
                                  'symbol': 'BTC',
                                  'amount': '0.1',
                                  'price': '50000'
                              })
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'No holdings of BTC' in data['error']
    
    def test_get_transactions(self, client, portfolio_tracker, sample_transaction_data):
        """Test getting transaction history"""
        # Save sample transactions
        with open(portfolio_tracker.transaction_file, 'w') as f:
            json.dump(sample_transaction_data, f)
        
        response = client.get('/api/portfolio/transactions')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'transactions' in data
        assert 'summary' in data
        assert len(data['transactions']) == 2
        assert data['summary']['total_buys'] == 2

class TestAPIRecommendations:
    """Test suite for recommendations endpoint"""
    
    def test_get_recommendations(self, client, temp_data_dir):
        """Test getting investment recommendations"""
        # Create sample recommendations
        recommendations = {
            'BTC': {
                'recommendation': 'BUY',
                'confidence': 0.75,
                'predicted_returns': {
                    '1_day': 2.5,
                    '7_day': 5.0,
                    '30_day': 15.0
                }
            }
        }
        
        os.makedirs(os.path.join(temp_data_dir, 'recommendations'), exist_ok=True)
        with open(os.path.join(temp_data_dir, 'recommendations', 'BTC_recommendations.json'), 'w') as f:
            json.dump(recommendations['BTC'], f)
        
        response = client.get('/api/recommendations')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'recommendations' in data
        assert 'BTC' in data['recommendations']
        assert data['recommendations']['BTC']['recommendation'] == 'BUY'

class TestAPISentiment:
    """Test suite for sentiment endpoints"""
    
    def test_get_sentiment_valid(self, client, temp_data_dir):
        """Test getting sentiment data"""
        # Create sample sentiment data
        import pandas as pd
        sentiment_data = pd.DataFrame({
            'date': pd.date_range(end='2024-01-10', periods=7),
            'sentiment_score': [65, 70, 68, 72, 69, 71, 73],
            'overall_sentiment': ['positive'] * 7
        })
        
        sentiment_data.to_csv(
            os.path.join(temp_data_dir, 'sentiment', 'BTC_daily_sentiment.csv'),
            index=False
        )
        
        response = client.get('/api/sentiment/BTC?days=5')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['symbol'] == 'BTC'
        assert len(data['sentiment']) == 5
        assert data['count'] == 5
    
    def test_get_sentiment_invalid_symbol(self, client):
        """Test getting sentiment for invalid symbol"""
        response = client.get('/api/sentiment/INVALID')
        
        assert response.status_code == 422
        data = json.loads(response.data)
        assert 'Invalid cryptocurrency symbol' in data['error']
    
    def test_get_sentiment_no_data(self, client):
        """Test getting sentiment when no data exists"""
        response = client.get('/api/sentiment/BTC')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'No sentiment data available' in data['error']

class TestAPIReports:
    """Test suite for report endpoints"""
    
    @patch('backend.portfolio_tracker.PortfolioTracker.generate_daily_report')
    def test_generate_report(self, mock_generate, client):
        """Test report generation"""
        mock_generate.return_value = '/path/to/report.html'
        
        response = client.post('/api/reports/generate')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'success'
        assert data['report_path'] == '/path/to/report.html'
        mock_generate.assert_called_once()

class TestAPIProcess:
    """Test suite for data processing endpoint"""
    
    @patch('backend.integration.run_data_processing')
    def test_process_data(self, mock_process, client):
        """Test data processing endpoint"""
        mock_process.return_value = None
        
        response = client.post('/api/process')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'success'
        assert 'Data processing completed' in data['message']
        mock_process.assert_called_once()