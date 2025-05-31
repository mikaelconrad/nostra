# Cryptocurrency Trading Simulator

An interactive cryptocurrency trading game that simulates real-world trading scenarios with AI-powered predictions, portfolio management, and comprehensive market analysis.

## Overview

This application provides an engaging way to learn cryptocurrency trading through a simulated environment. Trade Bitcoin (BTC) and Ethereum (ETH) using historical data, AI predictions, and technical analysis in a risk-free environment.

Key features:
- **Interactive Trading Game**: Simulate trading over customizable time periods (7-365 days)
- **AI Prediction Charts**: View 7, 14, and 30-day price predictions with visual forecasting
- **Social Sentiment Analysis**: Real-time Reddit sentiment monitoring and market mood tracking
- **Real Historical Data**: Trade using actual cryptocurrency price data
- **Portfolio Management**: Track your virtual portfolio performance in real-time
- **Sentiment-Based Trading Signals**: Get buy/sell recommendations based on community sentiment
- **Performance Analytics**: Comprehensive results and trading history analysis
- **Risk-Free Learning**: Practice trading strategies without financial risk

## System Architecture

The application consists of the following components:

1. **Backend**
   - Historical cryptocurrency data (CSV files)
   - AI prediction models for price forecasting
   - Social sentiment analysis engine with Reddit and market data integration
   - Game state management and portfolio tracking
   - Trading simulation engine with sentiment-based signals

2. **Frontend**
   - Interactive trading interface (Dash)
   - Real-time price charts and AI predictions (Plotly)
   - Social sentiment dashboard with live community data
   - Portfolio visualization and management
   - Game setup and results analysis

## Installation

### Prerequisites
- Python 3.10+
- pip package manager

### Setup

1. Clone the repository:
```
git clone <repository-url>
cd crypto_investment_app
```

2. Create and activate a virtual environment:
```
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```
pip install -r requirements.txt
```

4. Configure environment variables:
```
cp .env.example .env
# Edit .env with your API keys (see Setup Guide below)
```

5. Optional dependencies for enhanced functionality:
```
pip install kaleido  # For chart exports
python -c "import nltk; nltk.download('vader_lexicon')"  # For advanced sentiment analysis
```

## Usage

### Running the Trading Simulator

Start the game with a single command:

```bash
python run_game.py
```

The application will start and display:
```
==================================================
Starting Crypto Trading Simulator...
Open your browser to: http://localhost:8050
Press Ctrl+C to stop the server
==================================================
```

Open your browser and navigate to: **http://localhost:8050**

### Social Sentiment Analysis

The app now includes a comprehensive social sentiment analysis system:

**🔧 Quick Setup:**
1. Get Reddit API credentials from [Reddit Apps](https://www.reddit.com/prefs/apps)
2. Get CoinGecko API key from [CoinGecko API](https://www.coingecko.com/en/api) (optional)
3. Add credentials to your `.env` file
4. Run sentiment collection: `python run_sentiment_service.py once`

**📱 Features:**
- **Live Reddit Monitoring**: Real-time sentiment from r/Bitcoin, r/ethereum, r/CryptoCurrency
- **Sentiment Dashboard**: Visual gauges, trend charts, and community metrics
- **Trading Signals**: AI-powered buy/sell recommendations based on sentiment
- **Market Mood Tracking**: Fear & Greed Index and social volume indicators

**📊 Data Sources:**
- Reddit posts and comments with crypto-specific sentiment analysis
- CoinGecko community scores and social metrics
- Alternative.me Fear & Greed Index
- Real-time social volume and engagement tracking

See `SENTIMENT_SETUP.md` for detailed configuration instructions.

### How to Play

1. **Game Setup**
   - Set your initial cash amount (e.g., $10,000)
   - Choose starting cryptocurrency holdings (optional)
   - Select game duration (7-365 days)
   - Pick a start date from historical data

2. **Trading Phase**
   - View real-time price charts and AI predictions
   - Monitor your portfolio value and allocation
   - Execute buy/sell trades for BTC and ETH
   - Use AI prediction charts to inform your decisions
   - Advance through days to see market changes

3. **AI Predictions**
   - View 7, 14, and 30-day price forecasts
   - Charts show historical data + future predictions
   - Interactive dropdown to select prediction horizons
   - Visual indicators for predicted trends

4. **Game Completion**
   - Review your final portfolio performance
   - Analyze trading history and decisions
   - Compare against buy-and-hold strategies
   - Start new games to improve your skills

## Game Features

### Portfolio Management
- **Real-time Tracking**: Monitor your virtual portfolio value as markets change
- **Transaction History**: View all your buy/sell trades with timestamps
- **Performance Metrics**: Track gains/losses and compare to market performance
- **Asset Allocation**: Visual pie charts showing your BTC/ETH/Cash distribution

### AI-Powered Predictions
- **Multi-horizon Forecasting**: 7, 14, and 30-day price predictions
- **Visual Forecasting**: Charts showing historical data plus AI predictions
- **Interactive Analysis**: Select different prediction horizons to compare
- **Trend Indicators**: Visual cues for predicted price movements

### Trading Simulation
- **Realistic Market Data**: Uses actual historical cryptocurrency prices
- **Flexible Timeframes**: Choose game duration from 1 week to 1 year
- **Multiple Strategies**: Test different trading approaches risk-free
- **Performance Analytics**: Comprehensive results analysis at game end

## Configuration

You can customize the game by editing the `config.py` file:
- Default game duration and settings
- AI prediction parameters
- Chart display options
- Portfolio tracking settings

## Project Structure

```
crypto_investment_app/
├── backend/
│   ├── simple_data_collector.py     # Historical data loading
│   ├── neural_network_model.py      # AI prediction models
│   ├── portfolio_tracker.py         # Portfolio management
│   └── portfolio_tracker_db.py      # Database integration
├── frontend/
│   ├── app_game_simple.py          # Main trading game application
│   ├── game_state.py               # Game state management
│   ├── components/                 # UI components
│   └── validators.py               # Input validation
├── data/
│   ├── raw/                        # Historical price data (CSV)
│   │   ├── BTC_USD.csv
│   │   └── ETH_USD.csv
│   ├── portfolio.json              # Portfolio tracking
│   └── transactions.json           # Trading history
├── database/                       # Database models and management
├── config.py                       # Game configuration
├── run_game.py                     # Main application launcher
└── README.md
```

## Future Enhancements

Potential future improvements:
- **Additional Cryptocurrencies**: Support for more coins beyond BTC/ETH
- **Advanced AI Models**: More sophisticated prediction algorithms
- **Multiplayer Mode**: Compete against other traders in real-time
- **Custom Indicators**: Add your own technical analysis tools
- **Mobile Interface**: Responsive design for mobile trading
- **Leaderboards**: Track and compare performance with other players

## Quick Start

To get started immediately:

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Start the game**: `python run_game.py`
3. **Open browser**: Navigate to `http://localhost:8050`
4. **Begin trading**: Set up your game and start learning!

## Disclaimer

This is a trading simulation for educational purposes only. No real money is involved. The AI predictions are for demonstration and learning purposes and should not be used for actual cryptocurrency investment decisions.
