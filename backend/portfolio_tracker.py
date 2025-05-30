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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.logger import setup_logger
from utils.validators import CryptoValidator, FinancialValidator
from utils.error_handlers import ValidationError

# Set up logger
logger = setup_logger(__name__)

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
                return json.load(f)
        else:
            # Initialize empty portfolio
            return {
                'cash': config.INITIAL_INVESTMENT,
                'holdings': {},
                'last_update': datetime.now().isoformat()
            }
    
    def save_portfolio(self):
        """Save portfolio data to file"""
        self.portfolio['last_update'] = datetime.now().isoformat()
        with open(self.portfolio_file, 'w') as f:
            json.dump(self.portfolio, f, indent=4)
    
    def load_transactions(self):
        """Load transaction history from file"""
        if os.path.exists(self.transaction_file):
            with open(self.transaction_file, 'r') as f:
                return json.load(f)
        else:
            return []
    
    def save_transaction(self, transaction):
        """Save transaction to history"""
        transactions = self.load_transactions()
        transactions.append(transaction)
        with open(self.transaction_file, 'w') as f:
            json.dump(transactions, f, indent=4)
    
    def buy_crypto(self, symbol, amount, price):
        """Execute buy order for cryptocurrency"""
        try:
            # Validate inputs (already validated in API, but double-check)
            symbol = CryptoValidator.validate_symbol(symbol)
            amount = CryptoValidator.validate_amount(amount, symbol)
            price = CryptoValidator.validate_price(price)
            
            total_cost = amount * price
            
            # Validate sufficient funds
            if self.portfolio['cash'] < total_cost:
                return False, f"Insufficient funds. Available: CHF {self.portfolio['cash']:.2f}, Required: CHF {total_cost:.2f}"
            
            # Deduct cash
            self.portfolio['cash'] -= total_cost
            
            # Add to holdings
            if symbol not in self.portfolio['holdings']:
                self.portfolio['holdings'][symbol] = 0
            self.portfolio['holdings'][symbol] += amount
            
            # Save transaction
            transaction = {
                'type': 'buy',
                'symbol': symbol,
                'amount': amount,
                'price': price,
                'total': total_cost,
                'timestamp': datetime.now().isoformat()
            }
            self.save_transaction(transaction)
            
            # Save updated portfolio
            self.save_portfolio()
            
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
            if symbol not in self.portfolio['holdings']:
                return False, f"No holdings of {symbol} to sell"
            
            current_holdings = self.portfolio['holdings'][symbol]
            if current_holdings < amount:
                return False, f"Insufficient holdings. Available: {current_holdings} {symbol}, Requested: {amount} {symbol}"
            
            # Calculate proceeds
            total_proceeds = amount * price
            
            # Update holdings
            self.portfolio['holdings'][symbol] -= amount
            if self.portfolio['holdings'][symbol] < 0.00000001:  # Handle floating point precision
                del self.portfolio['holdings'][symbol]
            
            # Add cash
            self.portfolio['cash'] += total_proceeds
            
            # Save transaction
            transaction = {
                'type': 'sell',
                'symbol': symbol,
                'amount': amount,
                'price': price,
                'total': total_proceeds,
                'timestamp': datetime.now().isoformat()
            }
            self.save_transaction(transaction)
            
            # Save updated portfolio
            self.save_portfolio()
            
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
        total_value = self.portfolio['cash']
        
        for symbol, amount in self.portfolio['holdings'].items():
            if symbol in current_prices:
                total_value += amount * current_prices[symbol]
        
        return total_value
    
    def get_holdings_by_symbol(self):
        """Get detailed holdings by symbol"""
        holdings = {}
        
        # Add cash position
        holdings['cash'] = self.portfolio['cash']
        
        # Add crypto holdings
        for symbol, amount in self.portfolio['holdings'].items():
            holdings[symbol] = {
                'amount': amount,
                'value': 0  # Will be updated with current prices
            }
        
        return holdings
    
    def get_allocation(self, current_prices):
        """Calculate portfolio allocation percentages"""
        total_value = self.get_portfolio_value(current_prices)
        allocations = {}
        
        # Cash allocation
        allocations['cash'] = (self.portfolio['cash'] / total_value) * 100 if total_value > 0 else 0
        
        # Crypto allocations
        for symbol, amount in self.portfolio['holdings'].items():
            if symbol in current_prices:
                value = amount * current_prices[symbol]
                allocations[symbol] = (value / total_value) * 100 if total_value > 0 else 0
        
        return allocations
    
    def calculate_performance_metrics(self, current_prices):
        """Calculate portfolio performance metrics"""
        # Load transaction history
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
    
    def calculate_historical_performance(self, days=30):
        """Calculate historical portfolio performance"""
        # This is a simplified version - in practice, you'd track historical values
        # For now, we'll return mock data
        dates = pd.date_range(end=datetime.now(), periods=days)
        
        # Generate mock historical values
        initial_value = config.INITIAL_INVESTMENT
        values = []
        
        for i in range(days):
            # Simulate portfolio growth with some volatility
            growth_factor = 1 + (0.001 * i) + np.random.normal(0, 0.02)
            value = initial_value * growth_factor
            values.append(value)
        
        return pd.DataFrame({
            'date': dates,
            'value': values
        })
    
    def get_transaction_summary(self):
        """Get summary of all transactions"""
        transactions = self.load_transactions()
        
        summary = {
            'total_buys': 0,
            'total_sells': 0,
            'buy_volume': 0,
            'sell_volume': 0,
            'transactions': transactions[-10:]  # Last 10 transactions
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
        # Load current prices
        current_prices = {}
        for symbol in ['BTC', 'ETH', 'XRP']:
            price_file = os.path.join(config.RAW_DATA_DIRECTORY, f'{symbol}_USD.csv')
            if os.path.exists(price_file):
                df = pd.read_csv(price_file)
                if not df.empty:
                    current_prices[symbol] = df.iloc[-1]['Close']
        
        # Load recommendations
        recommendations = {}
        for symbol in ['BTC', 'ETH', 'XRP']:
            rec_file = os.path.join(config.RECOMMENDATIONS_DIRECTORY, f'{symbol}_recommendations.json')
            if os.path.exists(rec_file):
                with open(rec_file, 'r') as f:
                    recommendations[symbol] = json.load(f)
        
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
        
        # Generate HTML report
        html_content = self.generate_html_report(report_data)
        
        # Save HTML report
        report_filename = f"daily_report_{datetime.now().strftime('%Y%m%d')}.html"
        report_path = os.path.join(self.report_dir, report_filename)
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        print(f"Daily report generated: {report_path}")
        
        # Generate PDF if pdfkit is available
        if PDFKIT_AVAILABLE:
            pdf_path = report_path.replace('.html', '.pdf')
            try:
                pdfkit.from_file(report_path, pdf_path)
                print(f"PDF report generated: {pdf_path}")
            except Exception as e:
                print(f"Failed to generate PDF: {e}")
        
        return report_path
    
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
                    background-color: #f5f5f5;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }
                h1, h2 {
                    color: #333;
                }
                .metric-card {
                    background-color: #f8f9fa;
                    padding: 20px;
                    border-radius: 5px;
                    margin: 10px 0;
                }
                .positive {
                    color: #28a745;
                }
                .negative {
                    color: #dc3545;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }
                th, td {
                    padding: 10px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }
                th {
                    background-color: #f8f9fa;
                    font-weight: bold;
                }
                .recommendation {
                    padding: 10px;
                    margin: 10px 0;
                    border-radius: 5px;
                }
                .buy {
                    background-color: #d4edda;
                    border: 1px solid #c3e6cb;
                }
                .hold {
                    background-color: #fff3cd;
                    border: 1px solid #ffeeba;
                }
                .sell {
                    background-color: #f8d7da;
                    border: 1px solid #f5c6cb;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Cryptocurrency Investment Daily Report</h1>
                <p>Date: {{ data.date }}</p>
                
                <h2>Portfolio Overview</h2>
                <div class="metric-card">
                    <h3>Performance Metrics</h3>
                    <p>Total Invested: CHF {{ "%.2f"|format(data.metrics.total_invested) }}</p>
                    <p>Current Value: CHF {{ "%.2f"|format(data.metrics.current_value) }}</p>
                    <p class="{% if data.metrics.total_return >= 0 %}positive{% else %}negative{% endif %}">
                        Total Return: CHF {{ "%.2f"|format(data.metrics.total_return) }} 
                        ({{ "%.2f"|format(data.metrics.return_percentage) }}%)
                    </p>
                </div>
                
                <h2>Current Holdings</h2>
                <table>
                    <tr>
                        <th>Asset</th>
                        <th>Current Price</th>
                        <th>Holdings</th>
                        <th>Value (CHF)</th>
                        <th>Allocation</th>
                    </tr>
                    {% for symbol, holding in data.holdings.items() %}
                    <tr>
                        <td>{{ symbol }}</td>
                        <td>
                            {% if symbol == 'cash' %}
                                -
                            {% else %}
                                CHF {{ "%.2f"|format(data.current_prices[symbol]) }}
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
                    </tr>
                    {% endfor %}
                </table>
                
                <h2>Investment Recommendations</h2>
                {% for symbol, rec in data.recommendations.items() %}
                <div class="recommendation {{ rec.recommendation.lower() }}">
                    <h3>{{ symbol }}</h3>
                    <p><strong>Recommendation:</strong> {{ rec.recommendation }}</p>
                    <p><strong>Confidence:</strong> {{ "%.2f"|format(rec.confidence * 100) }}%</p>
                    <p><strong>Predicted Returns:</strong></p>
                    <ul>
                        <li>1 Day: {{ "%.2f"|format(rec.predicted_returns['1_day']) }}%</li>
                        <li>7 Days: {{ "%.2f"|format(rec.predicted_returns['7_day']) }}%</li>
                        <li>30 Days: {{ "%.2f"|format(rec.predicted_returns['30_day']) }}%</li>
                    </ul>
                </div>
                {% endfor %}
                
                <h2>Recent Transactions</h2>
                <table>
                    <tr>
                        <th>Date</th>
                        <th>Type</th>
                        <th>Symbol</th>
                        <th>Amount</th>
                        <th>Price</th>
                        <th>Total</th>
                    </tr>
                    {% for tx in data.transaction_summary.transactions %}
                    <tr>
                        <td>{{ tx.timestamp }}</td>
                        <td>{{ tx.type }}</td>
                        <td>{{ tx.symbol }}</td>
                        <td>{{ "%.6f"|format(tx.amount) }}</td>
                        <td>CHF {{ "%.2f"|format(tx.price) }}</td>
                        <td>CHF {{ "%.2f"|format(tx.total) }}</td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
        </body>
        </html>
        """
        
        # Create Jinja2 template
        template = jinja2.Template(template_str)
        
        # Render template with data
        html_content = template.render(data=data)
        
        return html_content

# Example usage
if __name__ == "__main__":
    tracker = PortfolioTracker()
    
    # Example: Buy some Bitcoin
    success, message = tracker.buy_crypto('BTC', 0.001, 50000)
    print(message)
    
    # Generate daily report
    report_path = tracker.generate_daily_report()
    print(f"Report generated: {report_path}")