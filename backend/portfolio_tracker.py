"""
Portfolio tracking and reporting module for cryptocurrency investment recommendation system
"""

import sys
import os
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import jinja2
try:
    import pdfkit
    PDFKIT_AVAILABLE = True
except ImportError:
    PDFKIT_AVAILABLE = False
    print("pdfkit module not available. PDF reports will not be generated.")

# Import configuration
sys.path.append('/home/ubuntu/crypto_investment_app')
import config

class PortfolioTracker:
    def __init__(self):
        """Initialize portfolio tracker"""
        self.portfolio_file = config.PORTFOLIO_FILE
        self.transaction_file = config.TRANSACTION_HISTORY_FILE
        self.report_dir = config.REPORT_DIRECTORY
        
        # Create report directory if it doesn't exist
        os.makedirs(self.report_dir, exist_ok=True)
        
        # Load portfolio data
        self.portfolio = self.load_portfolio()
        
    def load_portfolio(self):
        """Load portfolio data from file"""
        if os.path.exists(self.portfolio_file):
            with open(self.portfolio_file, 'r') as f:
                portfolio = json.load(f)
            return portfolio
        else:
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
            
            # Save default portfolio
            self.save_portfolio(portfolio)
            
            return portfolio
    
    def save_portfolio(self, portfolio=None):
        """Save portfolio data to file"""
        if portfolio is None:
            portfolio = self.portfolio
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.portfolio_file), exist_ok=True)
        
        with open(self.portfolio_file, 'w') as f:
            json.dump(portfolio, f, indent=4)
    
    def update_portfolio_values(self):
        """Update portfolio values based on current cryptocurrency prices"""
        # Load latest price data for each cryptocurrency
        for symbol in config.CRYPTO_SYMBOLS:
            if symbol in self.portfolio['holdings']:
                price_file = os.path.join(config.DATA_DIRECTORY, 'processed', f"{symbol.replace('-', '_')}_processed.csv")
                if os.path.exists(price_file):
                    df = pd.read_csv(price_file, index_col='date', parse_dates=True)
                    latest_price = df['close'].iloc[-1]
                    
                    # Update holding value
                    amount = self.portfolio['holdings'][symbol]['amount']
                    self.portfolio['holdings'][symbol]['value'] = amount * latest_price
        
        # Calculate total portfolio value
        total_value = self.portfolio['holdings']['cash']
        for symbol, data in self.portfolio['holdings'].items():
            if symbol != 'cash':
                total_value += data['value']
        
        self.portfolio['current_value'] = total_value
        
        # Save updated portfolio
        self.save_portfolio()
        
        return total_value
    
    def add_transaction(self, transaction_type, symbol, amount, price, fee):
        """
        Add a new transaction to the portfolio
        
        Args:
            transaction_type: 'buy' or 'sell'
            symbol: Cryptocurrency symbol (e.g., 'BTC-USD')
            amount: Amount of cryptocurrency
            price: Price per unit in USD
            fee: Transaction fee in CHF
            
        Returns:
            Success status and message
        """
        # Calculate transaction value in CHF
        value = amount * price
        
        # Update portfolio based on transaction type
        if transaction_type == "buy":
            # Check if enough cash is available
            if self.portfolio['holdings']['cash'] < value + fee:
                return False, "Not enough cash available for this transaction"
            
            # Update holdings
            if symbol not in self.portfolio['holdings']:
                self.portfolio['holdings'][symbol] = {'amount': 0, 'value': 0}
            
            self.portfolio['holdings'][symbol]['amount'] += amount
            self.portfolio['holdings'][symbol]['value'] += value
            self.portfolio['holdings']['cash'] -= (value + fee)
            
            # Add transaction to history
            transaction = {
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'type': 'buy',
                'symbol': symbol,
                'amount': amount,
                'price': price,
                'value': value,
                'fee': fee
            }
            self.portfolio['transactions'].append(transaction)
            
        elif transaction_type == "sell":
            # Check if enough cryptocurrency is available
            if symbol not in self.portfolio['holdings'] or self.portfolio['holdings'][symbol]['amount'] < amount:
                return False, "Not enough cryptocurrency available for this transaction"
            
            # Update holdings
            self.portfolio['holdings'][symbol]['amount'] -= amount
            self.portfolio['holdings'][symbol]['value'] = self.portfolio['holdings'][symbol]['amount'] * price
            self.portfolio['holdings']['cash'] += (value - fee)
            
            # Add transaction to history
            transaction = {
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'type': 'sell',
                'symbol': symbol,
                'amount': amount,
                'price': price,
                'value': -value,  # Negative value for sell transactions
                'fee': fee
            }
            self.portfolio['transactions'].append(transaction)
        
        # Save updated portfolio
        self.save_portfolio()
        
        return True, "Transaction added successfully"
    
    def calculate_portfolio_metrics(self):
        """Calculate portfolio performance metrics"""
        # Update portfolio values
        total_value = self.update_portfolio_values()
        
        # Calculate profit/loss
        profit_loss = total_value - self.portfolio['total_invested']
        profit_loss_pct = (profit_loss / self.portfolio['total_invested']) * 100 if self.portfolio['total_invested'] > 0 else 0
        
        # Calculate allocation percentages
        allocation = {}
        for symbol, data in self.portfolio['holdings'].items():
            if symbol == 'cash':
                allocation[symbol] = (data / total_value) * 100 if total_value > 0 else 0
            else:
                allocation[symbol] = (data['value'] / total_value) * 100 if total_value > 0 else 0
        
        # Calculate historical performance
        historical_performance = self.calculate_historical_performance()
        
        metrics = {
            'total_invested': self.portfolio['total_invested'],
            'current_value': total_value,
            'profit_loss': profit_loss,
            'profit_loss_pct': profit_loss_pct,
            'allocation': allocation,
            'historical_performance': historical_performance
        }
        
        return metrics
    
    def calculate_historical_performance(self):
        """Calculate historical portfolio performance"""
        transactions = self.portfolio['transactions']
        
        if not transactions:
            return []
        
        # Create DataFrame from transactions
        df = pd.DataFrame(transactions)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        # Resample by day and calculate cumulative portfolio value
        daily_value = df.resample('D').sum()
        daily_value['cumulative_value'] = daily_value['value'].cumsum() + config.INITIAL_INVESTMENT
        
        # Convert to list of dictionaries for reporting
        performance = []
        for date, row in daily_value.iterrows():
            performance.append({
                'date': date.strftime('%Y-%m-%d'),
                'value': row['cumulative_value']
            })
        
        return performance
    
    def generate_daily_report(self):
        """Generate daily investment report"""
        # Calculate portfolio metrics
        metrics = self.calculate_portfolio_metrics()
        
        # Load recommendations
        recommendations = {}
        for symbol in config.CRYPTO_SYMBOLS:
            base_symbol = symbol.split('-')[0]
            filename = os.path.join(config.DATA_DIRECTORY, 'recommendations', f"{base_symbol}_recommendations.json")
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    recommendations[symbol] = json.load(f)
        
        # Create report data
        report_data = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'portfolio': self.portfolio,
            'metrics': metrics,
            'recommendations': recommendations,
            'crypto_symbols': config.CRYPTO_SYMBOLS
        }
        
        # Generate HTML report
        html_report = self.generate_html_report(report_data)
        
        # Save HTML report
        report_filename = os.path.join(self.report_dir, f"daily_report_{datetime.now().strftime('%Y%m%d')}.html")
        with open(report_filename, 'w') as f:
            f.write(html_report)
        
        print(f"Daily report generated: {report_filename}")
        
        # Try to generate PDF report if pdfkit is available
        if PDFKIT_AVAILABLE:
            try:
                pdf_filename = os.path.join(self.report_dir, f"daily_report_{datetime.now().strftime('%Y%m%d')}.pdf")
                pdfkit.from_string(html_report, pdf_filename)
                print(f"PDF report generated: {pdf_filename}")
            except Exception as e:
                print(f"Could not generate PDF report: {str(e)}")
        
        return report_filename
    
    def generate_html_report(self, data):
        """Generate HTML report from template"""
        # Define HTML template
        template_str = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Cryptocurrency Investment Daily Report</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    color: #333;
                }
                .header {
                    text-align: center;
                    margin-bottom: 30px;
                }
                .section {
                    margin-bottom: 30px;
                }
                .summary-box {
                    display: flex;
                    justify-content: space-between;
                    margin-bottom: 20px;
                }
                .summary-item {
                    background-color: #f5f5f5;
                    padding: 15px;
                    border-radius: 5px;
                    width: 23%;
                    text-align: center;
                }
                .positive {
                    color: green;
                }
                .negative {
                    color: red;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                }
                th, td {
                    padding: 10px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }
                th {
                    background-color: #f2f2f2;
                }
                .recommendation {
                    display: flex;
                    margin-bottom: 20px;
                }
                .recommendation-card {
                    background-color: #f5f5f5;
                    padding: 15px;
                    border-radius: 5px;
                    margin-right: 15px;
                    width: 30%;
                }
                .buy {
                    border-left: 5px solid green;
                }
                .sell {
                    border-left: 5px solid red;
                }
                .hold {
                    border-left: 5px solid gray;
                }
                .strong {
                    font-weight: bold;
                }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Cryptocurrency Investment Daily Report</h1>
                <p>{{ data.date }}</p>
            </div>
            
            <div class="section">
                <h2>Portfolio Summary</h2>
                <div class="summary-box">
                    <div class="summary-item">
                        <h3>Total Invested</h3>
                        <p>CHF {{ "%.2f"|format(data.metrics.total_invested) }}</p>
                    </div>
                    <div class="summary-item">
                        <h3>Current Value</h3>
                        <p>CHF {{ "%.2f"|format(data.metrics.current_value) }}</p>
                    </div>
                    <div class="summary-item">
                        <h3>Profit/Loss</h3>
                        <p class="{{ 'positive' if data.metrics.profit_loss >= 0 else 'negative' }}">
                            CHF {{ "%.2f"|format(data.metrics.profit_loss) }}
                        </p>
                    </div>
                    <div class="summary-item">
                        <h3>Return</h3>
                        <p class="{{ 'positive' if data.metrics.profit_loss_pct >= 0 else 'negative' }}">
                            {{ "%.2f"|format(data.metrics.profit_loss_pct) }}%
                        </p>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>Holdings</h2>
                <table>
                    <tr>
                        <th>Asset</th>
                        <th>Amount</th>
                        <th>Value (CHF)</th>
                        <th>Allocation</th>
                    </tr>
                    {% for symbol, holding in data.portfolio.holdings.items() %}
                    <tr>
                        <td>
                            {% if symbol == 'cash' %}
                                Cash
                            {% else %}
                                {{ data.crypto_symbols[symbol] }} ({{ symbol }})
                            {% endif %}
                        </td>
                        <td>
                            {% if symbol == 'cash' %}
                                CHF {{ "%.2f"|format(holding) }}
                            {% else %}
                                {{ "%.6f"|format(holding.amount) }}
                            {% endif %}
                        </td>
                        <td>
                            {% if symbol == 'cash' %}
                                CHF {{ "%.2f"|format(holding) }}
                            {% else %}
                                CHF {{ "%.2f"|format(holding.value) }}
                            {% endif %}
                        </td>
                        <td>
                            {{ "%.2f"|format(data.metrics.allocation[symbol]) }}%
                        </td>
 <response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>