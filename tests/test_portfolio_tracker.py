"""
Tests for the portfolio tracker module
"""

import pytest
import json
import os
from datetime import datetime
from backend.portfolio_tracker import PortfolioTracker
from utils.error_handlers import ValidationError

class TestPortfolioTracker:
    """Test suite for PortfolioTracker class"""
    
    def test_initialization(self, portfolio_tracker):
        """Test portfolio tracker initialization"""
        assert portfolio_tracker is not None
        assert isinstance(portfolio_tracker.portfolio, dict)
        assert 'cash' in portfolio_tracker.portfolio
        assert 'holdings' in portfolio_tracker.portfolio
        assert portfolio_tracker.portfolio['cash'] == 1000.0  # Initial investment
    
    def test_load_empty_portfolio(self, portfolio_tracker, temp_data_dir):
        """Test loading portfolio when no file exists"""
        portfolio = portfolio_tracker.load_portfolio()
        assert portfolio['cash'] == 1000.0
        assert portfolio['holdings'] == {}
        assert 'last_update' in portfolio
    
    def test_save_and_load_portfolio(self, portfolio_tracker, temp_data_dir):
        """Test saving and loading portfolio data"""
        # Modify portfolio
        portfolio_tracker.portfolio['cash'] = 5000.0
        portfolio_tracker.portfolio['holdings'] = {'BTC': 0.1}
        
        # Save
        portfolio_tracker.save_portfolio()
        
        # Verify file exists
        assert os.path.exists(portfolio_tracker.portfolio_file)
        
        # Load and verify
        loaded = portfolio_tracker.load_portfolio()
        assert loaded['cash'] == 5000.0
        assert loaded['holdings']['BTC'] == 0.1
    
    def test_buy_crypto_success(self, portfolio_tracker):
        """Test successful cryptocurrency purchase"""
        # Buy 0.1 BTC at 50,000
        success, message = portfolio_tracker.buy_crypto('BTC', 0.1, 50000)
        
        assert success is True
        assert 'Bought 0.1 BTC' in message
        assert portfolio_tracker.portfolio['cash'] == -4000.0  # 1000 - 5000
        assert portfolio_tracker.portfolio['holdings']['BTC'] == 0.1
    
    def test_buy_crypto_insufficient_funds(self, portfolio_tracker):
        """Test buy with insufficient funds"""
        # Try to buy 1 BTC at 50,000 (need 50,000 but only have 1,000)
        success, message = portfolio_tracker.buy_crypto('BTC', 1.0, 50000)
        
        assert success is False
        assert 'Insufficient funds' in message
        assert portfolio_tracker.portfolio['cash'] == 1000.0  # Unchanged
        assert 'BTC' not in portfolio_tracker.portfolio['holdings']
    
    def test_buy_crypto_invalid_inputs(self, portfolio_tracker):
        """Test buy with invalid inputs"""
        # Invalid symbol
        success, message = portfolio_tracker.buy_crypto('INVALID', 0.1, 50000)
        assert success is False
        assert 'Invalid cryptocurrency symbol' in message
        
        # Negative amount
        success, message = portfolio_tracker.buy_crypto('BTC', -0.1, 50000)
        assert success is False
        assert 'must be positive' in message
        
        # Zero price
        success, message = portfolio_tracker.buy_crypto('BTC', 0.1, 0)
        assert success is False
        assert 'must be positive' in message
    
    def test_sell_crypto_success(self, portfolio_tracker):
        """Test successful cryptocurrency sale"""
        # First buy some BTC
        portfolio_tracker.portfolio['cash'] = 10000.0
        portfolio_tracker.buy_crypto('BTC', 0.2, 50000)
        
        # Now sell half
        success, message = portfolio_tracker.sell_crypto('BTC', 0.1, 55000)
        
        assert success is True
        assert 'Sold 0.1 BTC' in message
        assert portfolio_tracker.portfolio['cash'] == 5500.0  # 0 + 5500
        assert portfolio_tracker.portfolio['holdings']['BTC'] == 0.1
    
    def test_sell_crypto_insufficient_holdings(self, portfolio_tracker):
        """Test sell with insufficient holdings"""
        # Try to sell BTC without owning any
        success, message = portfolio_tracker.sell_crypto('BTC', 0.1, 50000)
        
        assert success is False
        assert 'No holdings of BTC' in message
        
        # Buy some and try to sell more
        portfolio_tracker.portfolio['cash'] = 10000.0
        portfolio_tracker.buy_crypto('BTC', 0.1, 50000)
        success, message = portfolio_tracker.sell_crypto('BTC', 0.2, 50000)
        
        assert success is False
        assert 'Insufficient holdings' in message
    
    def test_get_portfolio_value(self, portfolio_tracker, mock_current_prices):
        """Test portfolio value calculation"""
        # Set up portfolio
        portfolio_tracker.portfolio = {
            'cash': 1000.0,
            'holdings': {
                'BTC': 0.1,
                'ETH': 2.0
            }
        }
        
        value = portfolio_tracker.get_portfolio_value(mock_current_prices)
        expected = 1000.0 + (0.1 * 52900.0) + (2.0 * 2290.0)
        assert value == expected
    
    def test_get_allocation(self, portfolio_tracker, mock_current_prices):
        """Test portfolio allocation calculation"""
        # Set up portfolio
        portfolio_tracker.portfolio = {
            'cash': 1000.0,
            'holdings': {
                'BTC': 0.1,
                'ETH': 2.0
            }
        }
        
        allocations = portfolio_tracker.get_allocation(mock_current_prices)
        
        # Calculate expected allocations
        total_value = 1000.0 + (0.1 * 52900.0) + (2.0 * 2290.0)
        expected_cash = (1000.0 / total_value) * 100
        expected_btc = ((0.1 * 52900.0) / total_value) * 100
        expected_eth = ((2.0 * 2290.0) / total_value) * 100
        
        assert abs(allocations['cash'] - expected_cash) < 0.01
        assert abs(allocations['BTC'] - expected_btc) < 0.01
        assert abs(allocations['ETH'] - expected_eth) < 0.01
    
    def test_transaction_history(self, portfolio_tracker):
        """Test transaction history tracking"""
        # Make some transactions
        portfolio_tracker.portfolio['cash'] = 20000.0
        portfolio_tracker.buy_crypto('BTC', 0.1, 50000)
        portfolio_tracker.buy_crypto('ETH', 2.0, 2000)
        portfolio_tracker.sell_crypto('BTC', 0.05, 55000)
        
        # Load transactions
        transactions = portfolio_tracker.load_transactions()
        
        assert len(transactions) == 3
        assert transactions[0]['type'] == 'buy'
        assert transactions[0]['symbol'] == 'BTC'
        assert transactions[1]['type'] == 'buy'
        assert transactions[1]['symbol'] == 'ETH'
        assert transactions[2]['type'] == 'sell'
        assert transactions[2]['symbol'] == 'BTC'
    
    def test_transaction_summary(self, portfolio_tracker):
        """Test transaction summary generation"""
        # Make some transactions
        portfolio_tracker.portfolio['cash'] = 20000.0
        portfolio_tracker.buy_crypto('BTC', 0.1, 50000)
        portfolio_tracker.buy_crypto('ETH', 2.0, 2000)
        portfolio_tracker.sell_crypto('BTC', 0.05, 55000)
        
        summary = portfolio_tracker.get_transaction_summary()
        
        assert summary['total_buys'] == 2
        assert summary['total_sells'] == 1
        assert summary['buy_volume'] == 9000.0  # 5000 + 4000
        assert summary['sell_volume'] == 2750.0  # 0.05 * 55000
        assert len(summary['transactions']) == 3
    
    def test_calculate_performance_metrics(self, portfolio_tracker, mock_current_prices):
        """Test performance metrics calculation"""
        # Set up portfolio with some history
        portfolio_tracker.portfolio = {
            'cash': 1000.0,
            'holdings': {
                'BTC': 0.1,
                'ETH': 2.0
            }
        }
        
        # Create a transaction history
        transactions = [
            {
                'type': 'buy',
                'symbol': 'BTC',
                'amount': 0.1,
                'price': 45000.0,
                'total': 4500.0,
                'timestamp': datetime.now().isoformat()
            }
        ]
        with open(portfolio_tracker.transaction_file, 'w') as f:
            json.dump(transactions, f)
        
        metrics = portfolio_tracker.calculate_performance_metrics(mock_current_prices)
        
        assert 'total_invested' in metrics
        assert 'current_value' in metrics
        assert 'total_return' in metrics
        assert 'return_percentage' in metrics
        assert 'allocation' in metrics
        
        # Verify calculations
        assert metrics['total_invested'] == 1000.0  # Initial investment
        current_value = 1000.0 + (0.1 * 52900.0) + (2.0 * 2290.0)
        assert metrics['current_value'] == current_value
    
    def test_multiple_buy_same_crypto(self, portfolio_tracker):
        """Test multiple purchases of the same cryptocurrency"""
        portfolio_tracker.portfolio['cash'] = 20000.0
        
        # Buy BTC three times
        portfolio_tracker.buy_crypto('BTC', 0.1, 50000)
        portfolio_tracker.buy_crypto('BTC', 0.05, 52000)
        portfolio_tracker.buy_crypto('BTC', 0.02, 55000)
        
        # Check total holdings
        assert portfolio_tracker.portfolio['holdings']['BTC'] == 0.17
        assert portfolio_tracker.portfolio['cash'] == 11300.0  # 20000 - 5000 - 2600 - 1100
    
    def test_sell_all_holdings(self, portfolio_tracker):
        """Test selling all holdings of a cryptocurrency"""
        portfolio_tracker.portfolio['cash'] = 10000.0
        portfolio_tracker.buy_crypto('BTC', 0.1, 50000)
        
        # Sell all BTC
        success, message = portfolio_tracker.sell_crypto('BTC', 0.1, 55000)
        
        assert success is True
        assert 'BTC' not in portfolio_tracker.portfolio['holdings']
        assert portfolio_tracker.portfolio['cash'] == 10500.0  # 5000 + 5500