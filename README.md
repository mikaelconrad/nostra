# Crypto Trading Game

An interactive cryptocurrency trading simulator that allows users to practice trading Bitcoin and Ethereum with historical market data. Perfect for learning crypto trading strategies without risking real money.

## Features

### 🎮 Interactive Trading Simulation
- **Historical Data Trading**: Practice with real historical Bitcoin and Ethereum price data
- **Multiple Time Periods**: Choose from 30, 60, or 90-day trading simulations
- **Custom Date Ranges**: Select any historical period from 2020 onwards for your simulation
- **Real Market Conditions**: Experience actual market volatility and price movements

### 💼 Portfolio Management
- **Real-time Tracking**: Monitor your simulated portfolio performance during gameplay
- **Transaction History**: Complete record of all simulated buy/sell transactions
- **Performance Analytics**: Detailed metrics including returns, allocation, and profit/loss
- **Educational Reports**: Track your trading decisions and learn from outcomes

### 📊 Market Analysis Tools
- **Interactive Charts**: Candlestick charts with volume indicators and technical analysis
- **Historical Price Data**: Access to years of Bitcoin and Ethereum market data
- **Performance Benchmarks**: Compare your trading results against buy-and-hold strategies
- **Risk-Free Learning**: Master trading concepts without financial consequences

### 🔧 Technical Infrastructure
- **RESTful API**: Comprehensive API for game data and portfolio management
- **Modern Web Interface**: Dash-based interactive dashboard with real-time updates
- **Data Persistence**: JSON-based portfolio and transaction storage
- **Easy Setup**: Simple Python environment with minimal dependencies
- **Cross-platform**: Runs on Windows, macOS, and Linux

## Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/crypto-trading-game.git
   cd crypto-trading-game
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download market data**:
   ```bash
   # Collect historical price data
   python -m backend.data_collector
   ```

5. **Start the game**:
   ```bash
   python run_game.py
   ```

6. **Access the trading simulator**:
   - Game Interface: http://localhost:8062
   - API Health Check: http://localhost:5000/api/health

## How to Play

### Starting a Game
1. **Choose Game Length**: Select 30, 60, or 90 days, or set a custom duration
2. **Select Date Range**: Pick any historical period from 2020 onwards
3. **Set Starting Cash**: Begin with your chosen amount (default: 1000 CHF)
4. **Start Trading**: Use real historical data to make buy/sell decisions

### Trading Mechanics
```bash
# View current portfolio
http://localhost:8062  # Access the game interface

# Portfolio includes:
# - Cash balance
# - Cryptocurrency holdings (BTC, ETH)
# - Current portfolio value
# - Transaction history
```

### Game Features
- **Real Market Data**: Every price movement is based on actual historical data
- **Transaction Costs**: Realistic trading fees are applied to each transaction
- **Performance Tracking**: Monitor your gains/losses compared to market benchmarks
- **Learning Tools**: Understand market patterns and develop trading strategies

### After the Game
- **Results Analysis**: Review your trading performance and decisions
- **Strategy Comparison**: See how your active trading compared to holding
- **Learning Insights**: Identify successful patterns and areas for improvement

## Game API Reference

### Cryptocurrency Data
```bash
# Get historical price data
GET /api/crypto/{symbol}/data?days=30

# Update price data
POST /api/crypto/{symbol}/update
```

### Portfolio Management
```bash
# Get current portfolio
GET /api/portfolio

# Execute buy order
POST /api/portfolio/buy
{
  "symbol": "BTC",
  "amount": 0.001,
  "price": 45000
}

# Execute sell order
POST /api/portfolio/sell
{
  "symbol": "BTC",
  "amount": 0.001,
  "price": 46000
}

# Get transaction history
GET /api/portfolio/transactions
```

### Game Reports
```bash
# Generate performance report
POST /api/reports/generate

# System health check
GET /api/health
```

## Configuration

### Environment Variables
Create a `.env` file with the following configuration:

```env
# Application Settings
APP_ENV=development
DEBUG=true
FRONTEND_PORT=8062
API_PORT=5000

# Game Settings
INITIAL_INVESTMENT=1000
MONTHLY_CONTRIBUTION=500
TRANSACTION_FEE_PERCENTAGE=0.1
MIN_TRADE_AMOUNT=10

# Feature Flags
ENABLE_LIVE_TRADING=false
ENABLE_EMAIL_REPORTS=false
```

### Game Configuration
- **Starting Cash**: Customize initial portfolio balance
- **Transaction Fees**: Realistic trading costs (default: 0.1%)
- **Minimum Trade**: Minimum transaction amount (default: 10 CHF)
- **Supported Cryptocurrencies**: Bitcoin (BTC) and Ethereum (ETH)

## Architecture

### Game Components

```
├── Frontend (Dash)
│   ├── Trading Interface
│   ├── Portfolio Dashboard
│   ├── Historical Charts
│   └── Game Setup
│
├── Backend Services
│   ├── Data Collector
│   ├── Portfolio Tracker
│   └── Game Engine
│
├── API Layer
│   ├── RESTful Endpoints
│   ├── Game State Management
│   └── Transaction Processing
│
├── Data Storage
│   ├── Historical Price Data (CSV)
│   ├── Portfolio Data (JSON)
│   └── Transaction History
│
└── External APIs
    └── Yahoo Finance (for price data)
```

## Testing

Run the game test suite:

```bash
# Run all tests
python -m pytest

# Run specific test categories
python -m pytest tests/test_portfolio_tracker.py
python -m pytest tests/test_api.py
python -m pytest tests/test_validators.py

# Test game functionality
python test_game_interface.py

# Run with coverage
python -m pytest --cov=backend --cov=frontend --cov=api
```

## Deployment

### Local Development

```bash
# Start the trading game
python run_game.py

# The game will be available at:
# http://localhost:8062
```

### Production Deployment

1. **Environment Setup**:
   ```bash
   export APP_ENV=production
   export DEBUG=false
   export FRONTEND_PORT=8062
   export API_PORT=5000
   ```

2. **Data Preparation**:
   ```bash
   # Ensure historical data is available
   python -m backend.data_collector
   ```

3. **Start Services**:
   ```bash
   # Start the game server
   python run_game.py
   ```

## Contributing

We welcome contributions to improve the trading game!

### Development Setup

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Make changes and test**: `python -m pytest`
4. **Commit changes**: `git commit -m 'Add amazing feature'`
5. **Push to branch**: `git push origin feature/amazing-feature`
6. **Create Pull Request**

### Code Style
- Follow PEP 8 for Python code
- Use type hints where appropriate
- Add docstrings for public functions
- Include unit tests for new features

### Ideas for Contributions
- Additional cryptocurrencies (e.g., ADA, DOT, LINK)
- More sophisticated trading indicators
- Enhanced game scoring mechanisms
- Mobile-responsive interface improvements

## Educational Purpose

📚 **Learning Focus**: This trading simulator is designed for educational purposes to help users:
- Understand cryptocurrency market dynamics
- Practice trading strategies risk-free
- Learn about portfolio management
- Develop familiarity with market volatility
- Build confidence before real trading

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/your-username/crypto-trading-game/issues)
- **Feature Requests**: [GitHub Discussions](https://github.com/your-username/crypto-trading-game/discussions)

## Disclaimer

⚠️ **Important**: This is a simulation using historical data only. No real money is involved. This tool is for educational purposes and should not be considered as financial advice. Real cryptocurrency trading carries significant risk and can result in substantial losses. Always do your own research and consider consulting with a financial advisor before making actual investment decisions.