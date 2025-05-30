"""
Sentiment analysis module for cryptocurrency data using Twitter API
"""

import sys
import pandas as pd
import os
import json
from datetime import datetime, timedelta
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import re

# Import configuration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
        print("NLTK resources downloaded successfully")
    except Exception as e:
        print(f"Error downloading NLTK resources: {e}")

def search_twitter(query, count=100):
    """
    Mock Twitter search function
    In production, this would use the actual Twitter API
    
    Args:
        query: Search query string
        count: Number of tweets to fetch
        
    Returns:
        List of mock tweets
    """
    print(f"Mock searching Twitter for: {query}")
    
    # Generate mock tweets for demonstration
    tweets = []
    sentiments = ['positive', 'negative', 'neutral']
    
    for i in range(count):
        sentiment_type = sentiments[i % 3]
        
        if sentiment_type == 'positive':
            text = f"Love {query}! Great investment opportunity 🚀 #crypto"
        elif sentiment_type == 'negative':
            text = f"Worried about {query} performance. Market looks bearish 📉"
        else:
            text = f"Monitoring {query} closely. Interesting market movements today."
            
        tweets.append({
            'text': text,
            'created_at': (datetime.now() - timedelta(hours=i)).isoformat(),
            'user': f'user_{i}',
            'retweet_count': i * 10,
            'favorite_count': i * 15
        })
    
    return tweets

def clean_text(text):
    """Clean tweet text for sentiment analysis"""
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove mentions and hashtags
    text = re.sub(r'[@#]\w+', '', text)
    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def analyze_sentiment_vader(text):
    """Analyze sentiment using VADER"""
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(text)
    
    # Determine overall sentiment
    if scores['compound'] >= 0.05:
        sentiment = 'positive'
    elif scores['compound'] <= -0.05:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    
    return {
        'sentiment': sentiment,
        'compound': scores['compound'],
        'positive': scores['pos'],
        'negative': scores['neg'],
        'neutral': scores['neu']
    }

def analyze_sentiment_textblob(text):
    """Analyze sentiment using TextBlob"""
    blob = TextBlob(text)
    
    # Get polarity and subjectivity
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    # Determine sentiment
    if polarity > 0.1:
        sentiment = 'positive'
    elif polarity < -0.1:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    
    return {
        'sentiment': sentiment,
        'polarity': polarity,
        'subjectivity': subjectivity
    }

def analyze_crypto_sentiment(symbol, days=7):
    """
    Analyze sentiment for a specific cryptocurrency
    
    Args:
        symbol: Cryptocurrency symbol (e.g., 'BTC', 'ETH')
        days: Number of days to analyze
        
    Returns:
        DataFrame with sentiment analysis results
    """
    print(f"Analyzing sentiment for {symbol}...")
    
    # Download NLTK resources if needed
    download_nltk_resources()
    
    # Define search queries
    queries = [
        f"{symbol} cryptocurrency",
        f"${symbol} price",
        f"{symbol} investment",
        f"{symbol} trading"
    ]
    
    all_tweets = []
    
    # Collect tweets for each query
    for query in queries:
        tweets = search_twitter(query, count=25)
        all_tweets.extend(tweets)
    
    # Analyze sentiment for each tweet
    results = []
    
    for tweet in all_tweets:
        # Clean text
        cleaned_text = clean_text(tweet['text'])
        
        # Skip if text is too short
        if len(cleaned_text.split()) < 3:
            continue
        
        # Analyze with VADER
        vader_result = analyze_sentiment_vader(cleaned_text)
        
        # Analyze with TextBlob
        textblob_result = analyze_sentiment_textblob(cleaned_text)
        
        # Combine results
        result = {
            'timestamp': tweet['created_at'],
            'text': tweet['text'],
            'cleaned_text': cleaned_text,
            'vader_sentiment': vader_result['sentiment'],
            'vader_compound': vader_result['compound'],
            'textblob_sentiment': textblob_result['sentiment'],
            'textblob_polarity': textblob_result['polarity'],
            'retweet_count': tweet['retweet_count'],
            'favorite_count': tweet['favorite_count']
        }
        
        results.append(result)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Add date column
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    
    # Calculate daily sentiment
    daily_sentiment = df.groupby('date').agg({
        'vader_compound': 'mean',
        'textblob_polarity': 'mean',
        'vader_sentiment': lambda x: x.value_counts().index[0] if len(x) > 0 else 'neutral',
        'textblob_sentiment': lambda x: x.value_counts().index[0] if len(x) > 0 else 'neutral'
    }).reset_index()
    
    # Calculate overall sentiment score (0-100)
    daily_sentiment['sentiment_score'] = (
        (daily_sentiment['vader_compound'] + 1) * 25 +  # Scale from -1,1 to 0,50
        (daily_sentiment['textblob_polarity'] + 1) * 25  # Scale from -1,1 to 0,50
    )
    
    # Determine overall daily sentiment
    daily_sentiment['overall_sentiment'] = daily_sentiment['sentiment_score'].apply(
        lambda x: 'positive' if x > 60 else ('negative' if x < 40 else 'neutral')
    )
    
    return df, daily_sentiment

def save_sentiment_data(symbol, df_tweets, df_daily):
    """Save sentiment analysis results to CSV files"""
    # Save detailed tweet sentiment
    tweet_file = os.path.join(config.SENTIMENT_DATA_DIRECTORY, f'{symbol}_sentiment.csv')
    df_tweets.to_csv(tweet_file, index=False)
    print(f"Saved tweet sentiment data to: {tweet_file}")
    
    # Save daily sentiment summary
    daily_file = os.path.join(config.SENTIMENT_DATA_DIRECTORY, f'{symbol}_daily_sentiment.csv')
    df_daily.to_csv(daily_file, index=False)
    print(f"Saved daily sentiment data to: {daily_file}")

def main():
    """Main function to analyze sentiment for all cryptocurrencies"""
    # Create directories
    create_directories()
    
    # Analyze sentiment for each cryptocurrency
    for symbol in ['BTC', 'ETH', 'XRP']:
        print(f"\n{'='*50}")
        print(f"Analyzing {symbol} sentiment...")
        print(f"{'='*50}")
        
        # Analyze sentiment
        df_tweets, df_daily = analyze_crypto_sentiment(symbol, days=7)
        
        # Save results
        save_sentiment_data(symbol, df_tweets, df_daily)
        
        # Print summary
        print(f"\nSentiment Summary for {symbol}:")
        print(f"Total tweets analyzed: {len(df_tweets)}")
        print(f"Average sentiment score: {df_daily['sentiment_score'].mean():.2f}")
        print(f"Overall sentiment: {df_daily['overall_sentiment'].value_counts().to_dict()}")

if __name__ == "__main__":
    main()