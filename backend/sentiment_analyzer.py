"""
Sentiment analysis module for cryptocurrency data using Twitter API
"""

import sys
sys.path.append('/opt/.manus/.sandbox-runtime')
from data_api import ApiClient
import pandas as pd
import os
import json
from datetime import datetime, timedelta
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import re

# Import configuration
sys.path.append('/home/ubuntu/crypto_investment_app')
import config

def create_directories():
    """Create necessary directories for sentiment data storage"""
    os.makedirs(os.path.join(config.SENTIMENT_DATA_DIRECTORY), exist_ok=True)
    print(f"Created directories in {config.SENTIMENT_DATA_DIRECTORY}")

def download_nltk_resources():
    """Download required NLTK resources"""
    try:
        nltk.download('vader_lexicon')
        nltk.download('punkt')
        print("Downloaded NLTK resources")
    except Exception as e:
        print(f"Error downloading NLTK resources: {str(e)}")

def search_twitter(query, count=100):
    """
    Search Twitter for tweets matching the query
    
    Args:
        query: Search query string
        count: Number of tweets to return
        
    Returns:
        List of tweets
    """
    print(f"Searching Twitter for: {query}")
    
    try:
        client = ApiClient()
        result = client.call_api('Twitter/search_twitter', query={
            'query': query,
            'count': count,
            'type': 'Latest'
        })
        
        tweets = []
        
        # Extract tweets from the API response
        if 'result' in result and 'timeline' in result['result']:
            instructions = result['result']['timeline'].get('instructions', [])
            for instruction in instructions:
                entries = instruction.get('entries', [])
                for entry in entries:
                    content = entry.get('content', {})
                    items = content.get('items', [])
                    
                    for item in items:
                        item_content = item.get('item', {}).get('itemContent', {})
                        if 'tweet_results' in item_content:
                            tweet_result = item_content['tweet_results'].get('result', {})
                            if 'legacy' in tweet_result:
                                tweet_text = tweet_result['legacy'].get('full_text', '')
                                if not tweet_text:
                                    tweet_text = tweet_result['legacy'].get('text', '')
                                
                                tweet_data = {
                                    'text': tweet_text,
                                    'created_at': tweet_result['legacy'].get('created_at', ''),
                                    'user': tweet_result.get('core', {}).get('user_results', {}).get('result', {}).get('legacy', {}).get('screen_name', '')
                                }
                                tweets.append(tweet_data)
        
        print(f"Found {len(tweets)} tweets for query: {query}")
        return tweets
    
    except Exception as e:
        print(f"Error searching Twitter for {query}: {str(e)}")
        return []

def clean_tweet(tweet):
    """Clean tweet text by removing links, special characters, etc."""
    # Remove URLs
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
    # Remove user mentions
    tweet = re.sub(r'@\w+', '', tweet)
    # Remove hashtags
    tweet = re.sub(r'#\w+', '', tweet)
    # Remove RT
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # Remove special characters
    tweet = re.sub(r'[^\w\s]', '', tweet)
    # Remove extra spaces
    tweet = re.sub(r'\s+', ' ', tweet).strip()
    
    return tweet

def analyze_sentiment_vader(text):
    """
    Analyze sentiment using VADER
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary with sentiment scores
    """
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    
    return sentiment

def analyze_sentiment_textblob(text):
    """
    Analyze sentiment using TextBlob
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary with sentiment scores
    """
    analysis = TextBlob(text)
    
    return {
        'polarity': analysis.sentiment.polarity,
        'subjectivity': analysis.sentiment.subjectivity
    }

def analyze_tweets(tweets):
    """
    Analyze sentiment of tweets
    
    Args:
        tweets: List of tweet dictionaries
        
    Returns:
        DataFrame with sentiment analysis
    """
    results = []
    
    for tweet in tweets:
        text = tweet['text']
        cleaned_text = clean_tweet(text)
        
        if cleaned_text:
            vader_sentiment = analyze_sentiment_vader(cleaned_text)
            textblob_sentiment = analyze_sentiment_textblob(cleaned_text)
            
            result = {
                'text': text,
                'cleaned_text': cleaned_text,
                'created_at': tweet['created_at'],
                'user': tweet['user'],
                'vader_compound': vader_sentiment['compound'],
                'vader_pos': vader_sentiment['pos'],
                'vader_neu': vader_sentiment['neu'],
                'vader_neg': vader_sentiment['neg'],
                'textblob_polarity': textblob_sentiment['polarity'],
                'textblob_subjectivity': textblob_sentiment['subjectivity']
            }
            
            results.append(result)
    
    df = pd.DataFrame(results)
    
    # Convert created_at to datetime
    if 'created_at' in df.columns and not df.empty:
        df['created_at'] = pd.to_datetime(df['created_at'], format='%a %b %d %H:%M:%S %z %Y')
    
    return df

def collect_sentiment_data(crypto_symbol, keywords):
    """
    Collect and analyze sentiment data for a cryptocurrency
    
    Args:
        crypto_symbol: Cryptocurrency symbol (e.g., 'BTC')
        keywords: List of keywords to search for
        
    Returns:
        DataFrame with sentiment analysis
    """
    all_tweets = []
    
    for keyword in keywords:
        tweets = search_twitter(keyword)
        all_tweets.extend(tweets)
    
    # Remove duplicates
    unique_tweets = []
    seen_texts = set()
    
    for tweet in all_tweets:
        if tweet['text'] not in seen_texts:
            unique_tweets.append(tweet)
            seen_texts.add(tweet['text'])
    
    print(f"Collected {len(unique_tweets)} unique tweets for {crypto_symbol}")
    
    # Analyze sentiment
    sentiment_df = analyze_tweets(unique_tweets)
    
    return sentiment_df

def save_sentiment_data(df, crypto_symbol):
    """Save sentiment data to CSV file"""
    if df is not None and not df.empty:
        filename = os.path.join(config.SENTIMENT_DATA_DIRECTORY, f"{crypto_symbol}_sentiment.csv")
        df.to_csv(filename, index=False)
        print(f"Saved sentiment data to {filename}")
        return filename
    return None

def aggregate_daily_sentiment(df):
    """
    Aggregate sentiment data by day
    
    Args:
        df: DataFrame with sentiment data
        
    Returns:
        DataFrame with daily sentiment aggregation
    """
    if df is None or df.empty or 'created_at' not in df.columns:
        return None
    
    # Set created_at as index
    df = df.set_index('created_at')
    
    # Resample by day and calculate mean sentiment
    daily_sentiment = df.resample('D').agg({
        'vader_compound': 'mean',
        'vader_pos': 'mean',
        'vader_neu': 'mean',
        'vader_neg': 'mean',
        'textblob_polarity': 'mean',
        'textblob_subjectivity': 'mean'
    }).reset_index()
    
    # Convert timezone-aware datetime to naive datetime to match price data
    daily_sentiment['created_at'] = daily_sentiment['created_at'].dt.tz_localize(None)
    
    return daily_sentiment

def save_daily_sentiment(df, crypto_symbol):
    """Save daily sentiment data to CSV file"""
    if df is not None and not df.empty:
        filename = os.path.join(config.SENTIMENT_DATA_DIRECTORY, f"{crypto_symbol}_daily_sentiment.csv")
        df.to_csv(filename, index=False)
        print(f"Saved daily sentiment data to {filename}")
        return filename
    return None

def correlate_sentiment_with_price(sentiment_df, price_df, crypto_symbol):
    """
    Correlate sentiment data with price data
    
    Args:
        sentiment_df: DataFrame with daily sentiment data
        price_df: DataFrame with price data
        crypto_symbol: Cryptocurrency symbol
        
    Returns:
        DataFrame with correlation analysis
    """
    if sentiment_df is None or price_df is None or sentiment_df.empty or price_df.empty:
        return None
    
    try:
        # Ensure both DataFrames have date as index
        if 'created_at' in sentiment_df.columns:
            # Convert to datetime if it's not already
            sentiment_df['created_at'] = pd.to_datetime(sentiment_df['created_at'])
            # Make sure it's timezone naive
            if hasattr(sentiment_df['created_at'].dtype, 'tz') and sentiment_df['created_at'].dtype.tz is not None:
                sentiment_df['created_at'] = sentiment_df['created_at'].dt.tz_localize(None)
            sentiment_df = sentiment_df.set_index('created_at')
        
        # Make sure price_df index is timezone naive
        if hasattr(price_df.index.dtype, 'tz') and price_df.index.dtype.tz is not None:
            price_df.index = price_df.index.tz_localize(None)
        
        # Merge sentiment and price data
        merged_df = price_df.join(sentiment_df, how='inner')
        
        # Calculate correlation
        correlation = merged_df[['close', 'vader_compound', 'textblob_polarity']].corr()
        
        # Save correlation
        filename = os.path.join(config.SENTIMENT_DATA_DIRECTORY, f"{crypto_symbol}_sentiment_price_correlation.csv")
        correlation.to_csv(filename)
        print(f"Saved sentiment-price correlation to {filename}")
        
        return correlation
    except Exception as e:
        print(f"Error correlating sentiment with price: {str(e)}")
        return None

def create_daily_sentiment_update_script():
    """Create a script for daily sentiment updates"""
    script_content = """#!/bin/bash
# Daily update script for sentiment data

cd /home/ubuntu/crypto_investment_app
source venv/bin/activate
python backend/sentiment_analyzer.py

echo "Daily sentiment update completed at $(date)"
"""
    
    script_path = os.path.join('/home/ubuntu/crypto_investment_app', 'daily_sentiment_update.sh')
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make the script executable
    os.chmod(script_path, 0o755)
    
    print(f"Created daily sentiment update script at {script_path}")
    
    # Create a crontab entry instruction
    crontab_instruction = """
# To set up automatic daily sentiment updates, run the following command:
# (crontab -l 2>/dev/null; echo "0 1 * * * /home/ubuntu/crypto_investment_app/daily_sentiment_update.sh >> /home/ubuntu/crypto_investment_app/daily_sentiment_update.log 2>&1") | crontab -
# This will run the sentiment update script every day at 1 AM
"""
    
    instruction_path = os.path.join('/home/ubuntu/crypto_investment_app', 'sentiment_crontab_setup.txt')
    with open(instruction_path, 'w') as f:
        f.write(crontab_instruction)
    
    print(f"Created sentiment crontab setup instructions at {instruction_path}")

def load_price_data(symbol):
    """Load processed cryptocurrency price data"""
    filename = os.path.join(config.DATA_DIRECTORY, 'processed', f"{symbol.replace('-', '_')}_processed.csv")
    if os.path.exists(filename):
        df = pd.read_csv(filename, index_col='date', parse_dates=True)
        print(f"Loaded processed price data for {symbol}")
        return df
    else:
        print(f"Processed price data file not found: {filename}")
        return None

def main():
    """Main function to collect and analyze sentiment data"""
    create_directories()
    download_nltk_resources()
    
    for symbol, name in config.CRYPTO_SYMBOLS.items():
        # Get the base symbol (e.g., 'BTC' from 'BTC-USD')
        base_symbol = symbol.split('-')[0]
        
        # Get keywords for the cryptocurrency
        keywords = config.SENTIMENT_KEYWORDS.get(base_symbol, [name])
        
        # Collect and analyze sentiment data
        sentiment_df = collect_sentiment_data(base_symbol, keywords)
        
        if sentiment_df is not None and not sentiment_df.empty:
            # Save raw sentiment data
            save_sentiment_data(sentiment_df, base_symbol)
            
            # Aggregate daily sentiment
            daily_sentiment = aggregate_daily_sentiment(sentiment_df)
            
            if daily_sentiment is not None:
                # Save daily sentiment data
                save_daily_sentiment(daily_sentiment, base_symbol)
                
                # Load price data
                price_df = load_price_data(symbol)
                
                if price_df is not None:
                    # Correlate sentiment with price
                    correlate_sentiment_with_price(daily_sentiment, price_df, base_symbol)
    
    # Create daily sentiment update script
    create_daily_sentiment_update_script()

if __name__ == "__main__":
    main()
