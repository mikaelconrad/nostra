"""
Mock sentiment data generator for cryptocurrency sentiment analysis
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
import random

# Import configuration
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def generate_mock_tweets(crypto_symbol, num_tweets=100, days_back=30):
    """
    Generate mock tweets for a cryptocurrency
    
    Args:
        crypto_symbol: Cryptocurrency symbol (e.g., 'BTC')
        num_tweets: Number of tweets to generate
        days_back: Number of days to go back for tweet dates
        
    Returns:
        List of tweet dictionaries
    """
    print(f"Generating {num_tweets} mock tweets for {crypto_symbol}")
    
    # Positive and negative tweet templates
    positive_templates = [
        f"{crypto_symbol} looking bullish today! Price target ${{price}}",
        f"Just bought more {crypto_symbol}! Expecting a rally to ${{price}}",
        f"{crypto_symbol} technical analysis shows strong support. Hodl!",
        f"The future is bright for {crypto_symbol}. New ATH coming soon!",
        f"Institutional adoption of {crypto_symbol} is increasing. Bullish!",
        f"{crypto_symbol} fundamentals are stronger than ever. Buy the dip!",
        f"This {crypto_symbol} rally is just getting started! 🚀",
        f"Accumulating {crypto_symbol} at these levels is a no-brainer.",
        f"The {crypto_symbol} ecosystem is growing rapidly. Very optimistic!",
        f"Long-term {crypto_symbol} holders will be rewarded. Patience pays!"
    ]
    
    negative_templates = [
        f"{crypto_symbol} looking bearish. Might drop to ${{price}}",
        f"Just sold my {crypto_symbol}. Too much volatility for me.",
        f"{crypto_symbol} breaking key support levels. Be careful!",
        f"Regulatory concerns for {crypto_symbol} are mounting. Proceed with caution.",
        f"The {crypto_symbol} bubble is about to burst. Get out while you can!",
        f"Technical indicators show {crypto_symbol} is overbought. Correction incoming.",
        f"{crypto_symbol} volume declining. Not a good sign.",
        f"Whales dumping {crypto_symbol}. Price manipulation is real.",
        f"The {crypto_symbol} rally was just a dead cat bounce. More downside ahead.",
        f"Fundamentals don't support current {crypto_symbol} valuation. Overpriced!"
    ]
    
    neutral_templates = [
        f"{crypto_symbol} trading sideways today. Waiting for a breakout.",
        f"Monitoring {crypto_symbol} price action. No clear direction yet.",
        f"What's your price prediction for {crypto_symbol} this month?",
        f"{crypto_symbol} volatility has decreased recently. Consolidation phase?",
        f"Interesting developments in the {crypto_symbol} ecosystem.",
        f"New {crypto_symbol} update announced. Impact on price unclear.",
        f"Comparing {crypto_symbol} to other cryptocurrencies. Mixed results.",
        f"The {crypto_symbol} market is maturing. Expect less volatility.",
        f"Analyzing {crypto_symbol} on-chain metrics. Data is inconclusive.",
        f"Historical {crypto_symbol} patterns suggest we're in uncharted territory."
    ]
    
    # Generate random tweets
    tweets = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    for _ in range(num_tweets):
        # Randomly select sentiment
        sentiment = random.choice(['positive', 'negative', 'neutral'])
        
        # Select template based on sentiment
        if sentiment == 'positive':
            template = random.choice(positive_templates)
            price = random.randint(5000, 100000) if crypto_symbol == 'BTC' else random.randint(500, 10000) if crypto_symbol == 'ETH' else random.randint(1, 10)
        elif sentiment == 'negative':
            template = random.choice(negative_templates)
            price = random.randint(1000, 30000) if crypto_symbol == 'BTC' else random.randint(100, 3000) if crypto_symbol == 'ETH' else random.randint(0, 2)
        else:
            template = random.choice(neutral_templates)
            price = random.randint(3000, 50000) if crypto_symbol == 'BTC' else random.randint(300, 5000) if crypto_symbol == 'ETH' else random.randint(0, 5)
        
        # Format template with price
        text = template.replace('{{price}}', str(price))
        
        # Generate random date
        random_seconds = random.randint(0, int((end_date - start_date).total_seconds()))
        tweet_date = start_date + timedelta(seconds=random_seconds)
        
        # Format date string like Twitter's format
        date_str = tweet_date.strftime('%a %b %d %H:%M:%S +0000 %Y')
        
        # Create tweet dictionary
        tweet = {
            'text': text,
            'created_at': date_str,
            'user': f"crypto_user_{random.randint(1000, 9999)}"
        }
        
        tweets.append(tweet)
    
    return tweets

def analyze_sentiment(text):
    """
    Simple mock sentiment analysis function
    
    Args:
        text: Text to analyze
        
    Returns:
        Dictionary with sentiment scores
    """
    # Check for positive and negative keywords
    positive_keywords = ['bullish', 'rally', 'support', 'bright', 'increasing', 
                         'stronger', 'started', 'no-brainer', 'growing', 'rewarded']
    negative_keywords = ['bearish', 'sold', 'breaking', 'concerns', 'burst', 
                         'overbought', 'declining', 'dumping', 'bounce', 'overpriced']
    
    # Count occurrences
    positive_count = sum(1 for word in positive_keywords if word.lower() in text.lower())
    negative_count = sum(1 for word in negative_keywords if word.lower() in text.lower())
    
    # Calculate sentiment scores
    total = positive_count + negative_count
    if total == 0:
        compound = 0
        pos = 0.5
        neg = 0.5
        neu = 1.0
    else:
        pos = positive_count / (positive_count + negative_count)
        neg = negative_count / (positive_count + negative_count)
        compound = pos - neg
        neu = 1.0 - (pos + neg)
    
    return {
        'compound': compound,
        'pos': pos,
        'neg': neg,
        'neu': neu,
        'polarity': compound,
        'subjectivity': 0.5
    }

def generate_mock_sentiment_data():
    """
    Generate mock sentiment data for all cryptocurrencies
    """
    # Create sentiment data directory
    os.makedirs(config.SENTIMENT_DATA_DIRECTORY, exist_ok=True)
    
    for symbol, name in config.CRYPTO_SYMBOLS.items():
        # Get the base symbol (e.g., 'BTC' from 'BTC-USD')
        base_symbol = symbol.split('-')[0]
        
        # Generate mock tweets
        mock_tweets = generate_mock_tweets(base_symbol, num_tweets=200, days_back=60)
        
        # Analyze sentiment
        results = []
        for tweet in mock_tweets:
            text = tweet['text']
            sentiment = analyze_sentiment(text)
            
            result = {
                'text': text,
                'cleaned_text': text,
                'created_at': tweet['created_at'],
                'user': tweet['user'],
                'vader_compound': sentiment['compound'],
                'vader_pos': sentiment['pos'],
                'vader_neu': sentiment['neu'],
                'vader_neg': sentiment['neg'],
                'textblob_polarity': sentiment['polarity'],
                'textblob_subjectivity': sentiment['subjectivity']
            }
            
            results.append(result)
        
        # Create DataFrame
        sentiment_df = pd.DataFrame(results)
        
        # Convert created_at to datetime
        sentiment_df['created_at'] = pd.to_datetime(sentiment_df['created_at'], format='%a %b %d %H:%M:%S %z %Y')
        
        # Save raw sentiment data
        raw_filename = os.path.join(config.SENTIMENT_DATA_DIRECTORY, f"{base_symbol}_sentiment.csv")
        sentiment_df.to_csv(raw_filename, index=False)
        print(f"Saved sentiment data to {raw_filename}")
        
        # Aggregate daily sentiment
        sentiment_df = sentiment_df.set_index('created_at')
        daily_sentiment = sentiment_df.resample('D').agg({
            'vader_compound': 'mean',
            'vader_pos': 'mean',
            'vader_neu': 'mean',
            'vader_neg': 'mean',
            'textblob_polarity': 'mean',
            'textblob_subjectivity': 'mean'
        }).reset_index()
        
        # Convert timezone-aware datetime to naive datetime
        daily_sentiment['created_at'] = daily_sentiment['created_at'].dt.tz_localize(None)
        
        # Save daily sentiment data
        daily_filename = os.path.join(config.SENTIMENT_DATA_DIRECTORY, f"{base_symbol}_daily_sentiment.csv")
        daily_sentiment.to_csv(daily_filename, index=False)
        print(f"Saved daily sentiment data to {daily_filename}")
    
    print("Mock sentiment data generation completed")

if __name__ == "__main__":
    generate_mock_sentiment_data()
