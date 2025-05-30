"""
Database-backed portfolio tracking module for cryptocurrency investment recommendation system
"""

import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.logger import setup_logger
from utils.validators import CryptoValidator, FinancialValidator
from utils.error_handlers import ValidationError
from database.manager import db

# Set up logger
logger = setup_logger(__name__)

class PortfolioTrackerDB:
    """Database-backed portfolio tracker"""
    
    def __init__(self, user_id='default'):
        """Initialize portfolio tracker with database"""
        self.user_id = user_id
        self.db = db
        self._portfolio = None
        self._load_portfolio()
    
    def _load_portfolio(self):
        """Load portfolio from database"""
        self._portfolio = self.db.get_or_create_portfolio(self.user_id)
        logger.info(f"Loaded portfolio for user: {self.user_id}")
    
    @property
    def portfolio(self):
        """Get portfolio as dictionary (for compatibility)"""
        if not self._portfolio:
            self._load_portfolio()
        
        # Get holdings
        holdings = {}
        for holding in self._portfolio.holdings:
            holdings[holding.symbol] = holding.amount
        
        return {
            'cash': self._portfolio.cash,
            'holdings': holdings,
            'last_update': self._portfolio.last_update.isoformat() if self._portfolio.last_update else None
        }
    
    @property
    def portfolio_file(self):
        """Compatibility property"""
        return f"database://portfolio/{self.user_id}"
    
    @property
    def transaction_file(self):
        """Compatibility property"""
        return f"database://transactions/{self.user_id}"
    
    def save_portfolio(self):
        """Save portfolio to database (compatibility method)"""
        # In DB version, saves are automatic
        logger.debug("Portfolio auto-saved to database")
    
    def load_transactions(self):
        """Load transaction history from database"""
        return self.db.get_transactions(self._portfolio.id)
    
    def save_transaction(self, transaction):
        """Save transaction to database (compatibility method)"""
        # Transaction is saved in buy/sell methods
        logger.debug("Transaction saved to database")
    
    def buy_crypto(self, symbol, amount, price):
        """Execute buy order for cryptocurrency"""
        try:
            # Validate inputs
            symbol = CryptoValidator.validate_symbol(symbol)
            amount = CryptoValidator.validate_amount(amount, symbol)
            price = CryptoValidator.validate_price(price)
            
            total_cost = amount * price
            
            # Check sufficient funds
            if self._portfolio.cash < total_cost:
                return False, f"Insufficient funds. Available: CHF {self._portfolio.cash:.2f}, Required: CHF {total_cost:.2f}"
            
            # Update cash in database
            new_cash = self._portfolio.cash - total_cost
            self.db.update_portfolio_cash(self._portfolio.id, new_cash)
            
            # Update holdings in database
            current_holding = next((h for h in self._portfolio.holdings if h.symbol == symbol), None)
            current_amount = current_holding.amount if current_holding else 0
            new_amount = current_amount + amount
            self.db.update_holding(self._portfolio.id, symbol, new_amount)
            
            # Save transaction
            transaction_data = {
                'type': 'buy',
                'symbol': symbol,
                'amount': amount,
                'price': price,
                'total': total_cost,
                'timestamp': datetime.utcnow()
            }
            self.db.add_transaction(self._portfolio.id, transaction_data)
            
            # Reload portfolio to get updated data
            self._load_portfolio()
            
            logger.info(f"Buy order executed: {amount} {symbol} at CHF {price:.2f}")
            return True, f"Bought {amount} {symbol} at CHF {price:.2f}"
            
        except ValidationError as e:
            logger.error(f"Validation error in buy_crypto: {str(e)}")
            return False, str(e)
        except Exception as e:
            logger.error(f"Error in buy_crypto: {str(e)}", exc_info=True)
            return False, "Failed to execute buy order"
    
    def sell_crypto(self, symbol, amount, price):
        """Execute sell order for cryptocurrency"""
        try:
            # Validate inputs
            symbol = CryptoValidator.validate_symbol(symbol)
            amount = CryptoValidator.validate_amount(amount, symbol)
            price = CryptoValidator.validate_price(price)
            
            # Check holdings
            current_holding = next((h for h in self._portfolio.holdings if h.symbol == symbol), None)
            
            if not current_holding:
                return False, f"No holdings of {symbol} to sell"
            
            if current_holding.amount < amount:
                return False, f"Insufficient holdings. Available: {current_holding.amount} {symbol}, Requested: {amount} {symbol}"
            
            # Calculate proceeds
            total_proceeds = amount * price
            
            # Update holdings in database
            new_amount = current_holding.amount - amount
            self.db.update_holding(self._portfolio.id, symbol, new_amount)
            
            # Update cash in database
            new_cash = self._portfolio.cash + total_proceeds
            self.db.update_portfolio_cash(self._portfolio.id, new_cash)
            
            # Save transaction
            transaction_data = {
                'type': 'sell',
                'symbol': symbol,
                'amount': amount,
                'price': price,
                'total': total_proceeds,
                'timestamp': datetime.utcnow()
            }
            self.db.add_transaction(self._portfolio.id, transaction_data)
            
            # Reload portfolio to get updated data
            self._load_portfolio()
            
            logger.info(f"Sell order executed: {amount} {symbol} at CHF {price:.2f}")
            return True, f"Sold {amount} {symbol} at CHF {price:.2f}"
            
        except ValidationError as e:
            logger.error(f"Validation error in sell_crypto: {str(e)}")
            return False, str(e)
        except Exception as e:
            logger.error(f"Error in sell_crypto: {str(e)}", exc_info=True)
            return False, "Failed to execute sell order"
    
    def get_portfolio_value(self, current_prices):
        """Calculate total portfolio value"""
        total_value = self._portfolio.cash
        
        for holding in self._portfolio.holdings:
            if holding.symbol in current_prices:
                total_value += holding.amount * current_prices[holding.symbol]
        
        return total_value
    
    def get_holdings_by_symbol(self):
        """Get detailed holdings by symbol"""
        holdings = {'cash': self._portfolio.cash}
        
        for holding in self._portfolio.holdings:
            holdings[holding.symbol] = {
                'amount': holding.amount,
                'value': 0  # Will be updated with current prices
            }
        
        return holdings
    
    def get_allocation(self, current_prices):
        """Calculate portfolio allocation percentages"""
        total_value = self.get_portfolio_value(current_prices)
        allocations = {}
        
        # Cash allocation
        allocations['cash'] = (self._portfolio.cash / total_value) * 100 if total_value > 0 else 0
        
        # Crypto allocations
        for holding in self._portfolio.holdings:
            if holding.symbol in current_prices:
                value = holding.amount * current_prices[holding.symbol]
                allocations[holding.symbol] = (value / total_value) * 100 if total_value > 0 else 0
        
        return allocations
    
    def calculate_performance_metrics(self, current_prices):
        """Calculate portfolio performance metrics"""
        # Get transaction history
        transactions = self.load_transactions()
        
        # Calculate total invested
        total_invested = config.INITIAL_INVESTMENT
        
        # Add monthly contributions based on transaction history
        if transactions:
            first_transaction_date = datetime.fromisoformat(transactions[0]['timestamp'])
            months_active = (datetime.now() - first_transaction_date).days // 30
            total_invested += months_active * config.MONTHLY_CONTRIBUTION
        
        # Calculate current value
        current_value = self.get_portfolio_value(current_prices)
        
        # Calculate returns
        total_return = current_value - total_invested
        return_percentage = (total_return / total_invested) * 100 if total_invested > 0 else 0
        
        # Calculate allocation
        allocations = self.get_allocation(current_prices)
        
        return {
            'total_invested': total_invested,
            'current_value': current_value,
            'total_return': total_return,
            'return_percentage': return_percentage,
            'allocation': allocations
        }
    
    def get_transaction_summary(self):
        """Get summary of all transactions"""
        transactions = self.load_transactions()
        
        summary = {
            'total_buys': 0,
            'total_sells': 0,
            'buy_volume': 0,
            'sell_volume': 0,
            'transactions': transactions[:10]  # Last 10 transactions
        }
        
        for tx in transactions:
            if tx['type'] == 'buy':
                summary['total_buys'] += 1
                summary['buy_volume'] += tx['total']
            else:
                summary['total_sells'] += 1
                summary['sell_volume'] += tx['total']
        
        return summary
    
    def generate_daily_report(self):
        """Generate daily investment report"""
        # Get current prices from database
        current_prices = {}
        for symbol in ['BTC', 'ETH', 'XRP']:
            price = self.db.get_latest_price(symbol)
            if price:
                current_prices[symbol] = price
        
        # Get recommendations from database
        recommendations = self.db.get_latest_recommendations()
        
        # Calculate performance metrics
        metrics = self.calculate_performance_metrics(current_prices)
        
        # Get holdings
        holdings = self.get_holdings_by_symbol()
        for symbol in holdings:
            if symbol != 'cash' and symbol in current_prices:
                holdings[symbol]['value'] = holdings[symbol]['amount'] * current_prices[symbol]
        
        # Prepare report data
        report_data = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'current_prices': current_prices,
            'holdings': holdings,
            'metrics': metrics,
            'recommendations': recommendations,
            'transaction_summary': self.get_transaction_summary()
        }
        
        # For now, return the data (HTML generation can be added later)
        logger.info("Daily report generated from database")
        return report_data

# Alias for compatibility
PortfolioTracker = PortfolioTrackerDB