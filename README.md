# Cryptocurrency Investment Recommendation System

A comprehensive application for making data-driven cryptocurrency investment decisions using neural networks, historical price data, and sentiment analysis.

## Overview

This system helps you make informed investment decisions for Bitcoin (BTC), Ethereum (ETH), and Ripple (XRP) cryptocurrencies. It uses historical price data and sentiment analysis to generate investment recommendations for 1-day, 7-day, and 30-day horizons.

Key features:
- Historical cryptocurrency price data collection and analysis
- Technical indicator calculation (SMA, EMA, RSI, MACD, Bollinger Bands)
- Social media sentiment analysis
- Neural network prediction model using LSTM architecture
- Portfolio tracking and performance reporting
- Interactive dashboard with visualizations
- Daily investment recommendations

## System Architecture

The application consists of the following components:

1. **Backend**
   - Data collection module (Yahoo Finance API)
   - Data processing module (technical indicators)
   - Sentiment analysis module
   - Neural network prediction model (TensorFlow LSTM)
   - Portfolio tracking and reporting

2. **Frontend**
   - Interactive dashboard (Dash)
   - Price charts and visualizations (Plotly)
   - Portfolio management interface
   - Investment recommendations display

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

4. Optional dependencies for enhanced functionality:
```
pip install kaleido  # For chart exports
```

## Usage

### Initial Setup

1. Run the integration script to set up the system:
```
python backend/integration.py
```

This will:
- Collect historical cryptocurrency data
- Process data and calculate technical indicators
- Generate mock sentiment data
- Create initial portfolio
- Set up daily updates

### Running the Application

1. Start the application:
```
python frontend/app.py
```

2. Open your browser and navigate to:
```
http://localhost:8050
```

### Daily Updates

The system includes a script for daily updates. To set up automatic daily updates:

1. Review the crontab setup instructions:
```
cat crontab_setup.txt
```

2. Follow the instructions to set up a cron job that will run the daily update script.

## Portfolio Management

The system allows you to:
- Track your cryptocurrency investments
- Record buy/sell transactions
- Monitor portfolio performance
- View allocation across different cryptocurrencies
- Generate daily investment reports

## Investment Strategy

The system provides investment recommendations based on:
- Historical price patterns
- Technical indicators
- Market sentiment
- Neural network predictions

Recommendations include:
- Action (BUY, SELL, HOLD)
- Strength (STRONG, MODERATE, NEUTRAL)
- Expected return percentage
- Predicted price

## Configuration

You can customize the system by editing the `config.py` file:
- Initial investment amount
- Monthly contribution
- Cryptocurrencies to track
- Neural network parameters
- Reporting settings

## Project Structure

```
crypto_investment_app/
├── backend/
│   ├── data_collector.py
│   ├── data_processor.py
│   ├── mock_sentiment_generator.py
│   ├── neural_network_model.py
│   ├── portfolio_tracker.py
│   └── integration.py
├── frontend/
│   └── app.py
├── data/
│   ├── raw/
│   ├── processed/
│   ├── sentiment/
│   └── recommendations/
├── models/
├── reports/
├── config.py
├── daily_update.sh
└── README.md
```

## Future Enhancements

Potential future improvements:
- Real Twitter/X API integration for sentiment analysis
- Additional cryptocurrencies
- More sophisticated neural network architectures
- Automated trading capabilities
- Mobile application
- Email notifications for investment recommendations

## Disclaimer

This system is for educational and research purposes only. Always conduct your own research before making investment decisions. Cryptocurrency investments are subject to high market risk.
