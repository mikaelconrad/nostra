# Social Sentiment Analysis Setup Guide

This guide explains how to set up and use the new social sentiment analysis feature in the Crypto Trading Simulator.

## Overview

The sentiment analysis feature adds a comprehensive social media sentiment monitoring system that includes:

- **Reddit Sentiment Analysis**: Real-time monitoring of cryptocurrency discussions
- **Market Sentiment Data**: Integration with CoinGecko and market data sources  
- **AI-Powered Analysis**: Advanced sentiment scoring with crypto-specific terminology
- **Trading Signals**: Sentiment-based trading recommendations
- **Interactive Dashboard**: Visual sentiment displays and trend analysis

## Quick Start (Demo Mode)

The sentiment panel will work immediately with mock data for demonstration purposes:

```bash
# Install new dependencies
pip install praw beautifulsoup4 vaderSentiment

# Run the game with sentiment features
python run_game.py
```

The sentiment panel will appear below the "AI Market Predictions" section with:
- 📊 Sentiment gauge showing current market mood
- 📈 7-day sentiment trend chart  
- 📰 Top Reddit posts carousel
- 📊 Community metrics and statistics
- 🎯 Trading signals based on sentiment

## Full Setup (Live Data)

### Step 1: Reddit API Setup

1. **Create Reddit Application**:
   - Go to https://www.reddit.com/prefs/apps
   - Click "Create App" or "Create Another App"
   - Choose "script" as the app type
   - Fill in the form:
     - Name: `CryptoSentimentCollector`
     - Description: `Collecting cryptocurrency sentiment data`
     - About URL: (leave blank)
     - Redirect URI: `http://localhost:8080`

2. **Get API Credentials**:
   - Note down the client ID (under the app name)
   - Note down the client secret
   - Set environment variables:
   ```bash
   export REDDIT_CLIENT_ID="your_client_id"
   export REDDIT_CLIENT_SECRET="your_client_secret"
   export REDDIT_USER_AGENT="CryptoSentimentCollector/1.0"
   ```

### Step 2: Market Data API Setup (Optional)

1. **CoinGecko API** (Recommended):
   - Free tier: 10-50 calls/minute
   - Sign up at: https://www.coingecko.com/en/api
   - Set environment variable:
   ```bash
   export COINGECKO_API_KEY="your_api_key"
   ```

2. **CoinMarketCap API** (Optional):
   - Free tier: 333 calls/month
   - Sign up at: https://coinmarketcap.com/api/
   - Set environment variable:
   ```bash
   export COINMARKETCAP_API_KEY="your_api_key"
   ```

### Step 3: Install Dependencies

```bash
# Install required Python packages
pip install -r requirements.txt

# Download NLTK data (for enhanced sentiment analysis)
python -c "import nltk; nltk.download('vader_lexicon')"
```

### Step 4: Run Data Collection Service

Start the background service to collect sentiment data:

```bash
# Run one-time collection for testing
python run_sentiment_service.py once

# Run continuous collection service
python run_sentiment_service.py
```

The service will:
- Collect Reddit posts from r/Bitcoin, r/ethereum, r/CryptoCurrency
- Gather market sentiment from CoinGecko  
- Analyze sentiment using AI models
- Cache data for fast retrieval
- Run scheduled collections throughout the day

## API Endpoints

The sentiment feature adds several new API endpoints:

### Current Data
- `GET /api/sentiment/social/reddit/current` - Current Reddit sentiment
- `GET /api/sentiment/social/market/current` - Current market sentiment  
- `GET /api/sentiment/social/aggregated?coin=BTC` - Aggregated daily sentiment

### Historical Data
- `GET /api/sentiment/social/history?days=7&coin=BTC` - Historical sentiment
- `GET /api/sentiment/social/trends?coin=BTC` - Sentiment trends and analysis

### Trading Signals
- `GET /api/sentiment/social/signals?coin=BTC` - Sentiment-based trading signals
- `GET /api/sentiment/social/correlation?coin=BTC` - Sentiment vs price correlation

### Cache Management
- `GET /api/sentiment/cache/stats` - Cache statistics
- `POST /api/sentiment/cache/cleanup` - Clean up old cache data

## Game Features

### Sentiment Panel Components

1. **Sentiment Gauge**: Visual indicator of current market sentiment
2. **Trend Chart**: 7-day sentiment history with bullish/bearish zones
3. **Reddit Highlights**: Top posts from cryptocurrency subreddits
4. **Community Metrics**: Post counts, social volume, Fear & Greed Index

### Trading Integration

1. **Sentiment Signals**: 
   - Strong Buy/Sell signals based on sentiment thresholds
   - Confidence levels and risk assessments
   - Strategy recommendations

2. **Game Challenges**:
   - Sentiment Accuracy Challenge: Make profitable trades using sentiment signals
   - Contrarian Trader Challenge: Successfully trade against sentiment

3. **Performance Tracking**:
   - Track sentiment-based trading accuracy
   - Compare sentiment strategy returns
   - View sentiment trade history

## Configuration

### Data Collection Settings

Edit `run_sentiment_service.py` to customize collection:

```python
config = {
    'reddit_collection_interval': 'daily',  # 'hourly', 'daily', 'manual'
    'market_collection_interval': 'daily',
    'max_retries': 3,
    'retry_delay': 300,  # 5 minutes
    'primary_subreddits_only': True,  # Focus on main subreddits
    'debug_mode': False
}
```

### Cache Management

- Data is automatically cached for fast retrieval
- Files older than 30 days are compressed
- Files older than 365 days are removed
- Manual cleanup: `POST /api/sentiment/cache/cleanup`

## Data Sources

### Reddit Subreddits

**Primary** (higher priority):
- r/Bitcoin - Main Bitcoin community
- r/ethereum - Main Ethereum community  
- r/CryptoCurrency - General crypto discussions

**Secondary**:
- r/btc - Alternative Bitcoin community
- r/ethfinance - Ethereum finance discussions
- r/CryptoMarkets - Trading-focused discussions

### Market Data Sources

- **CoinGecko**: Community scores, social metrics, developer activity
- **Alternative.me**: Fear & Greed Index
- **CoinMarketCap**: Community sentiment (web scraping)

## Sentiment Analysis Features

### Crypto-Specific Analysis

- **Terminology Recognition**: Bitcoin, Ethereum, DeFi, HODL, diamond hands, etc.
- **Meme Detection**: Moon, lambo, rocket emojis, "this is the way"
- **Price Predictions**: Extract price targets from posts
- **Confidence Scoring**: Assess reliability of sentiment indicators

### Multi-Method Analysis

1. **Crypto Lexicon**: Custom dictionary of crypto sentiment words
2. **TextBlob**: General sentiment analysis (if available)
3. **NLTK VADER**: Compound sentiment scoring (if available)
4. **Combined Scoring**: Weighted combination of all methods

## Troubleshooting

### Common Issues

1. **No Sentiment Data**: 
   - Check Reddit API credentials
   - Verify internet connection
   - Run one-time collection: `python run_sentiment_service.py once`

2. **Rate Limiting**:
   - Reddit: 60 requests/minute limit
   - CoinGecko: 10-50 calls/minute (free tier)
   - Service automatically handles rate limiting

3. **Missing Dependencies**:
   ```bash
   pip install praw beautifulsoup4 vaderSentiment dash-bootstrap-components
   python -c "import nltk; nltk.download('vader_lexicon')"
   ```

### Service Health Check

Check if the sentiment service is running properly:

```bash
# Check service status
curl http://localhost:5000/api/sentiment/cache/stats

# View recent Reddit data
curl http://localhost:5000/api/sentiment/social/reddit/current

# Check aggregated sentiment
curl http://localhost:5000/api/sentiment/social/aggregated?coin=BTC
```

## File Structure

```
backend/social_sentiment/
├── __init__.py
├── schemas.py                    # Data schemas for sentiment data
├── cache_manager.py              # Data caching and storage
├── sentiment_analyzer.py         # AI sentiment analysis engine
├── data_collection_service.py    # Background collection service
├── reddit/
│   ├── __init__.py
│   └── reddit_collector.py       # Reddit API integration
└── market_data/
    ├── __init__.py
    └── market_collector.py       # Market data collection

frontend/components/
├── sentiment_panel.py            # Main sentiment UI panel
└── sentiment_signals.py          # Trading signals and game features

data/social_sentiment/
├── daily/                        # Daily collection data
├── cache/                        # Cache metadata
└── aggregated/                   # Processed sentiment data

run_sentiment_service.py          # Sentiment collection service runner
SENTIMENT_SETUP.md               # This setup guide
```

## Next Steps

1. **Run the Demo**: Start with mock data to see the UI
2. **Set up Reddit API**: Get real Reddit sentiment data
3. **Configure Collection**: Set up the background service
4. **Monitor Performance**: Track sentiment-based trading results
5. **Customize Analysis**: Adjust sentiment scoring for your needs

For advanced usage and customization, see the docstrings in each module and the API endpoint documentation above.