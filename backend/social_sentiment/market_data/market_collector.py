"""
Market sentiment data collector for CoinMarketCap and CoinGecko
"""
import requests
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os
from bs4 import BeautifulSoup
import json

from ..schemas import MarketSentiment, save_sentiment_data
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from utils.logger import setup_logger

logger = setup_logger(__name__)

class MarketDataCollector:
    """Collects market sentiment data from various sources"""
    
    def __init__(self):
        """Initialize market data collector"""
        # Load environment variables from .env file
        from dotenv import load_dotenv
        load_dotenv()
        
        # API keys and URLs
        self.coingecko_api_key = os.getenv('COINGECKO_API_KEY')
        self.coingecko_api_url = os.getenv('COINGECKO_API_URL', 'https://api.coingecko.com/api/v3')
        self.coinmarketcap_api_key = os.getenv('COINMARKETCAP_API_KEY')
        self.coinmarketcap_api_url = os.getenv('COINMARKETCAP_API_URL', 'https://pro-api.coinmarketcap.com/v1')
        self.alternative_api_url = os.getenv('ALTERNATIVE_API_URL', 'https://api.alternative.me/fng')
        
        # Rate limiting
        self.coingecko_rate_limit = 10  # calls per minute for free tier
        self.coinmarketcap_rate_limit = 333  # calls per month for free tier
        
        # Request delays
        self.coingecko_delay = 60 / self.coingecko_rate_limit
        self.coinmarketcap_delay = 1  # Conservative delay
        
        # Last request times
        self.last_coingecko_request = 0
        self.last_coinmarketcap_request = 0
        
        # User agents for web scraping
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        ]
        
        # Coin mappings
        self.coin_mappings = {
            'BTC': {'coingecko_id': 'bitcoin', 'coinmarketcap_id': '1'},
            'ETH': {'coingecko_id': 'ethereum', 'coinmarketcap_id': '1027'}
        }
    
    def _rate_limit_coingecko(self):
        """Rate limiting for CoinGecko API"""
        current_time = time.time()
        time_since_last = current_time - self.last_coingecko_request
        
        if time_since_last < self.coingecko_delay:
            sleep_time = self.coingecko_delay - time_since_last
            time.sleep(sleep_time)
        
        self.last_coingecko_request = time.time()
    
    def _rate_limit_coinmarketcap(self):
        """Rate limiting for CoinMarketCap API"""
        current_time = time.time()
        time_since_last = current_time - self.last_coinmarketcap_request
        
        if time_since_last < self.coinmarketcap_delay:
            sleep_time = self.coinmarketcap_delay - time_since_last
            time.sleep(sleep_time)
        
        self.last_coinmarketcap_request = time.time()
    
    def collect_coingecko_sentiment(self, coin: str) -> Optional[MarketSentiment]:
        """
        Collect sentiment data from CoinGecko API
        
        Args:
            coin: Cryptocurrency symbol ('BTC', 'ETH')
            
        Returns:
            MarketSentiment object or None if failed
        """
        if coin not in self.coin_mappings:
            logger.error(f"Unsupported coin: {coin}")
            return None
        
        coin_id = self.coin_mappings[coin]['coingecko_id']
        
        try:
            self._rate_limit_coingecko()
            
            # Use API endpoint with proper key
            url = f"{self.coingecko_api_url}/coins/{coin_id}"
            params = {
                'localization': 'false',
                'tickers': 'false',
                'market_data': 'true',
                'community_data': 'true',
                'developer_data': 'true',
                'sparkline': 'false'
            }
            
            headers = {}
            if self.coingecko_api_key:
                headers['x-cg-pro-api-key'] = self.coingecko_api_key
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract sentiment indicators
            market_data = data.get('market_data', {})
            community_data = data.get('community_data', {})
            developer_data = data.get('developer_data', {})
            
            sentiment = MarketSentiment(
                source='coingecko',
                coin=coin,
                timestamp=datetime.now(),
                community_score=community_data.get('community_score'),
                social_volume=community_data.get('twitter_followers', 0) + community_data.get('reddit_subscribers', 0),
                developer_activity=developer_data.get('commit_count_4_weeks'),
                sentiment_score=community_data.get('community_score', 0) / 100.0 if community_data.get('community_score') else None,
                price_usd=market_data.get('current_price', {}).get('usd'),
                volume_24h=market_data.get('total_volume', {}).get('usd'),
                market_cap=market_data.get('market_cap', {}).get('usd')
            )
            
            # Calculate sentiment label based on score
            if sentiment.sentiment_score:
                if sentiment.sentiment_score > 0.6:
                    sentiment.sentiment_label = 'bullish'
                elif sentiment.sentiment_score < 0.4:
                    sentiment.sentiment_label = 'bearish'
                else:
                    sentiment.sentiment_label = 'neutral'
            
            logger.info(f"Collected CoinGecko sentiment for {coin}")
            return sentiment
            
        except Exception as e:
            logger.error(f"Error collecting CoinGecko sentiment for {coin}: {e}")
            return None
    
    def scrape_coinmarketcap_sentiment(self, coin: str) -> Optional[MarketSentiment]:
        """
        Scrape sentiment data from CoinMarketCap website
        
        Args:
            coin: Cryptocurrency symbol ('BTC', 'ETH')
            
        Returns:
            MarketSentiment object or None if failed
        """
        if coin not in self.coin_mappings:
            logger.error(f"Unsupported coin: {coin}")
            return None
        
        try:
            self._rate_limit_coinmarketcap()
            
            # Get coin page URL
            coin_name = 'bitcoin' if coin == 'BTC' else 'ethereum'
            url = f"https://coinmarketcap.com/currencies/{coin_name}/"
            
            headers = {
                'User-Agent': self.user_agents[0],
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Try to extract sentiment indicators from the page
            # Note: This is fragile and may break if CMC changes their layout
            sentiment_indicators = self._extract_cmc_sentiment_indicators(soup)
            
            sentiment = MarketSentiment(
                source='coinmarketcap',
                coin=coin,
                timestamp=datetime.now(),
                **sentiment_indicators
            )
            
            logger.info(f"Scraped CoinMarketCap sentiment for {coin}")
            return sentiment
            
        except Exception as e:
            logger.error(f"Error scraping CoinMarketCap sentiment for {coin}: {e}")
            return None
    
    def _extract_cmc_sentiment_indicators(self, soup: BeautifulSoup) -> Dict:
        """
        Extract sentiment indicators from CoinMarketCap page
        
        Args:
            soup: BeautifulSoup object of the page
            
        Returns:
            Dictionary of sentiment indicators
        """
        indicators = {}
        
        try:
            # Try to find social metrics (this is very fragile)
            # Look for community score or social dominance
            
            # Extract price data
            price_element = soup.find('span', {'class': lambda x: x and 'price' in x.lower()})
            if price_element:
                price_text = price_element.get_text().replace('$', '').replace(',', '')
                try:
                    indicators['price_usd'] = float(price_text)
                except ValueError:
                    pass
            
            # Extract market cap
            market_cap_element = soup.find('dd', {'class': lambda x: x and 'market-cap' in x.lower()})
            if market_cap_element:
                mc_text = market_cap_element.get_text().replace('$', '').replace(',', '').replace('B', '000000000').replace('M', '000000')
                try:
                    indicators['market_cap'] = float(mc_text)
                except ValueError:
                    pass
            
            # Try to extract sentiment score (very basic)
            # Look for positive/negative indicators in the page
            page_text = soup.get_text().lower()
            positive_words = ['bullish', 'positive', 'growth', 'rising', 'increase']
            negative_words = ['bearish', 'negative', 'decline', 'falling', 'decrease']
            
            positive_count = sum(1 for word in positive_words if word in page_text)
            negative_count = sum(1 for word in negative_words if word in page_text)
            
            if positive_count + negative_count > 0:
                sentiment_score = positive_count / (positive_count + negative_count)
                indicators['sentiment_score'] = sentiment_score
                
                if sentiment_score > 0.6:
                    indicators['sentiment_label'] = 'bullish'
                elif sentiment_score < 0.4:
                    indicators['sentiment_label'] = 'bearish'
                else:
                    indicators['sentiment_label'] = 'neutral'
        
        except Exception as e:
            logger.error(f"Error extracting CMC sentiment indicators: {e}")
        
        return indicators
    
    def collect_fear_greed_index(self) -> Optional[float]:
        """
        Collect Fear & Greed Index from alternative.me
        
        Returns:
            Fear & Greed Index value (0-100) or None if failed
        """
        try:
            url = f"{self.alternative_api_url}/"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data.get('data') and len(data['data']) > 0:
                fng_value = int(data['data'][0]['value'])
                logger.info(f"Collected Fear & Greed Index: {fng_value}")
                return fng_value
            
        except Exception as e:
            logger.error(f"Error collecting Fear & Greed Index: {e}")
        
        return None
    
    def collect_daily_market_sentiment(self, coins: List[str] = None) -> Dict:
        """
        Collect daily market sentiment data for specified coins
        
        Args:
            coins: List of coin symbols (defaults to ['BTC', 'ETH'])
            
        Returns:
            Dictionary with collected market sentiment data
        """
        if coins is None:
            coins = ['BTC', 'ETH']
        
        market_data = []
        
        logger.info(f"Starting daily market sentiment collection for {coins}")
        
        # Collect Fear & Greed Index
        fear_greed = self.collect_fear_greed_index()
        
        for coin in coins:
            try:
                # Try CoinGecko first (more reliable API)
                sentiment = self.collect_coingecko_sentiment(coin)
                if sentiment:
                    sentiment.fear_greed_index = fear_greed
                    market_data.append(sentiment.to_dict())
                
                # Add delay between coins
                time.sleep(2)
                
                # Optionally try CoinMarketCap scraping as backup
                # (commented out to avoid being too aggressive)
                # cmc_sentiment = self.scrape_coinmarketcap_sentiment(coin)
                # if cmc_sentiment:
                #     market_data.append(cmc_sentiment.to_dict())
                
            except Exception as e:
                logger.error(f"Error collecting market sentiment for {coin}: {e}")
                continue
        
        data = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'fear_greed_index': fear_greed,
            'market_sentiment': market_data,
            'collection_metadata': {
                'collected_at': datetime.now().isoformat(),
                'coins': coins,
                'total_records': len(market_data)
            }
        }
        
        logger.info(f"Market sentiment collection complete: {len(market_data)} records")
        
        return data
    
    def save_daily_market_data(self, data: Dict, base_path: str = "data/social_sentiment/daily"):
        """
        Save daily market sentiment data to file
        
        Args:
            data: Data dictionary from collect_daily_market_sentiment
            base_path: Base directory for saving files
        """
        date_str = data['date']
        filename = f"market_{date_str}.json"
        filepath = os.path.join(base_path, filename)
        
        # Create directory if it doesn't exist
        os.makedirs(base_path, exist_ok=True)
        
        save_sentiment_data(data, filepath)
        logger.info(f"Saved daily market sentiment data to {filepath}")
        
        return filepath

# Configuration instructions
MARKET_DATA_SETUP_INSTRUCTIONS = """
Market Data API Setup (Optional):

1. CoinGecko API (Recommended):
   - Free tier: 10-50 calls/minute
   - Sign up at: https://www.coingecko.com/en/api
   - Set environment variable: COINGECKO_API_KEY="your_api_key"

2. CoinMarketCap API (Optional):
   - Free tier: 333 calls/month
   - Sign up at: https://coinmarketcap.com/api/
   - Set environment variable: COINMARKETCAP_API_KEY="your_api_key"

Note: The collector will work without API keys using web scraping and free endpoints,
but rate limits will be more restrictive.
"""