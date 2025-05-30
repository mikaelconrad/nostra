"""
Game state management for the Crypto Trading Simulation
Handles all game-related data and state transitions
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
from dataclasses import dataclass, asdict
from enum import Enum

from config import GameState, MIN_CASH_AMOUNT, TRANSACTION_FEE_PERCENTAGE


@dataclass
class Portfolio:
    """Represents a portfolio state at a specific point in time"""
    cash: float
    btc_amount: float
    eth_amount: float
    
    def get_total_value(self, btc_price: float, eth_price: float) -> float:
        """Calculate total portfolio value in CHF"""
        return self.cash + (self.btc_amount * btc_price) + (self.eth_amount * eth_price)
    
    def to_dict(self) -> Dict:
        """Convert portfolio to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Portfolio':
        """Create portfolio from dictionary"""
        return cls(**data)


@dataclass
class Transaction:
    """Represents a single transaction"""
    date: str
    type: str  # 'buy' or 'sell'
    crypto: str  # 'BTC' or 'ETH'
    amount: float
    price: float
    total_value: float
    fee: float
    
    def to_dict(self) -> Dict:
        """Convert transaction to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Transaction':
        """Create transaction from dictionary"""
        return cls(**data)


class SimulationGame:
    """Main game state manager"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset game to initial state"""
        self.state = GameState.SETUP
        self.start_date: Optional[str] = None
        self.end_date: Optional[str] = None
        self.current_date: Optional[str] = None
        self.initial_portfolio: Optional[Portfolio] = None
        self.current_portfolio: Optional[Portfolio] = None
        self.transactions: List[Transaction] = []
        self.daily_values: List[Dict] = []  # Track portfolio value over time
        self.current_prices: Dict[str, float] = {}
        
    def initialize_game(self, start_date: str, duration_days: int, 
                       initial_cash: float, initial_btc: float, initial_eth: float) -> bool:
        """Initialize a new game with given parameters"""
        try:
            # Validate inputs
            if initial_cash < MIN_CASH_AMOUNT:
                return False
            
            # Set dates
            self.start_date = start_date
            self.current_date = start_date
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            self.end_date = (start_dt + timedelta(days=duration_days)).strftime('%Y-%m-%d')
            
            # Initialize portfolio
            self.initial_portfolio = Portfolio(
                cash=initial_cash,
                btc_amount=initial_btc,
                eth_amount=initial_eth
            )
            self.current_portfolio = Portfolio(
                cash=initial_cash,
                btc_amount=initial_btc,
                eth_amount=initial_eth
            )
            
            # Change state
            self.state = GameState.PLAYING
            
            return True
            
        except Exception:
            return False
    
    def advance_day(self) -> bool:
        """Advance to the next trading day"""
        if self.state != GameState.PLAYING:
            return False
        
        current_dt = datetime.strptime(self.current_date, '%Y-%m-%d')
        next_dt = current_dt + timedelta(days=1)
        self.current_date = next_dt.strftime('%Y-%m-%d')
        
        # Check if game is complete
        if self.current_date >= self.end_date:
            self.state = GameState.COMPLETED
        
        return True
    
    def execute_trade(self, trade_type: str, crypto: str, amount: float, price: float) -> Tuple[bool, str]:
        """
        Execute a buy or sell trade
        Returns: (success, message)
        """
        if self.state != GameState.PLAYING:
            return False, "Game is not in playing state"
        
        # Calculate transaction details
        total_value = amount * price
        fee = total_value * (TRANSACTION_FEE_PERCENTAGE / 100)
        total_cost = total_value + fee if trade_type == 'buy' else total_value - fee
        
        # Validate trade
        if trade_type == 'buy':
            if self.current_portfolio.cash < total_cost:
                return False, "Insufficient cash"
            
            # Execute buy
            self.current_portfolio.cash -= total_cost
            if crypto == 'BTC':
                self.current_portfolio.btc_amount += amount
            else:
                self.current_portfolio.eth_amount += amount
                
        else:  # sell
            # Check if user has enough crypto
            if crypto == 'BTC' and self.current_portfolio.btc_amount < amount:
                return False, "Insufficient BTC"
            elif crypto == 'ETH' and self.current_portfolio.eth_amount < amount:
                return False, "Insufficient ETH"
            
            # Execute sell
            self.current_portfolio.cash += total_value - fee
            if crypto == 'BTC':
                self.current_portfolio.btc_amount -= amount
            else:
                self.current_portfolio.eth_amount -= amount
        
        # Record transaction
        transaction = Transaction(
            date=self.current_date,
            type=trade_type,
            crypto=crypto,
            amount=amount,
            price=price,
            total_value=total_value,
            fee=fee
        )
        self.transactions.append(transaction)
        
        return True, "Trade executed successfully"
    
    def record_daily_value(self, btc_price: float, eth_price: float):
        """Record the portfolio value for the current day"""
        self.current_prices = {'BTC': btc_price, 'ETH': eth_price}
        
        total_value = self.current_portfolio.get_total_value(btc_price, eth_price)
        initial_value = self.initial_portfolio.get_total_value(btc_price, eth_price)
        
        self.daily_values.append({
            'date': self.current_date,
            'total_value': total_value,
            'cash': self.current_portfolio.cash,
            'btc_value': self.current_portfolio.btc_amount * btc_price,
            'eth_value': self.current_portfolio.eth_amount * eth_price,
            'profit_loss': total_value - initial_value,
            'profit_loss_pct': ((total_value - initial_value) / initial_value) * 100
        })
    
    def get_performance_metrics(self) -> Dict:
        """Calculate performance metrics for the game"""
        if not self.daily_values:
            return {}
        
        initial_value = self.daily_values[0]['total_value']
        final_value = self.daily_values[-1]['total_value']
        
        # Calculate metrics
        total_return = final_value - initial_value
        total_return_pct = (total_return / initial_value) * 100
        
        # Calculate max drawdown
        peak = self.daily_values[0]['total_value']
        max_drawdown = 0
        for day in self.daily_values:
            if day['total_value'] > peak:
                peak = day['total_value']
            drawdown = (peak - day['total_value']) / peak * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Count trades
        total_trades = len(self.transactions)
        buy_trades = len([t for t in self.transactions if t.type == 'buy'])
        sell_trades = len([t for t in self.transactions if t.type == 'sell'])
        total_fees = sum(t.fee for t in self.transactions)
        
        return {
            'initial_value': initial_value,
            'final_value': final_value,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'buy_trades': buy_trades,
            'sell_trades': sell_trades,
            'total_fees': total_fees,
            'days_played': len(self.daily_values)
        }
    
    def to_dict(self) -> Dict:
        """Serialize game state to dictionary"""
        return {
            'state': self.state,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'current_date': self.current_date,
            'initial_portfolio': self.initial_portfolio.to_dict() if self.initial_portfolio else None,
            'current_portfolio': self.current_portfolio.to_dict() if self.current_portfolio else None,
            'transactions': [t.to_dict() for t in self.transactions],
            'daily_values': self.daily_values,
            'current_prices': self.current_prices
        }
    
    def from_dict(self, data: Dict):
        """Load game state from dictionary"""
        self.state = data.get('state', GameState.SETUP)
        self.start_date = data.get('start_date')
        self.end_date = data.get('end_date')
        self.current_date = data.get('current_date')
        
        if data.get('initial_portfolio'):
            self.initial_portfolio = Portfolio.from_dict(data['initial_portfolio'])
        
        if data.get('current_portfolio'):
            self.current_portfolio = Portfolio.from_dict(data['current_portfolio'])
        
        self.transactions = [Transaction.from_dict(t) for t in data.get('transactions', [])]
        self.daily_values = data.get('daily_values', [])
        self.current_prices = data.get('current_prices', {})
    
    def save_to_json(self, filepath: str):
        """Save game state to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def load_from_json(self, filepath: str) -> bool:
        """Load game state from JSON file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            self.from_dict(data)
            return True
        except Exception:
            return False


# Global game instance
game_instance = SimulationGame()