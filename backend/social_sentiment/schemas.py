"""
Data schemas for social sentiment analysis
"""
from typing import Dict, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass, asdict
import json

@dataclass
class RedditPost:
    """Schema for Reddit post data"""
    id: str
    title: str
    content: str
    subreddit: str
    author: str
    score: int
    upvote_ratio: float
    num_comments: int
    created_utc: float
    url: str
    
    # Sentiment analysis fields
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[str] = None  # 'bullish', 'bearish', 'neutral'
    confidence: Optional[float] = None
    
    # Crypto-specific fields
    mentioned_coins: Optional[List[str]] = None
    price_predictions: Optional[List[Dict]] = None
    
    # Processing metadata
    processed_at: Optional[datetime] = None
    relevance_score: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        if self.processed_at:
            data['processed_at'] = self.processed_at.isoformat()
        return data

@dataclass
class RedditComment:
    """Schema for Reddit comment data"""
    id: str
    post_id: str
    body: str
    author: str
    score: int
    created_utc: float
    
    # Sentiment analysis fields
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[str] = None
    confidence: Optional[float] = None
    
    # Processing metadata
    processed_at: Optional[datetime] = None
    relevance_score: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        if self.processed_at:
            data['processed_at'] = self.processed_at.isoformat()
        return data

@dataclass
class MarketSentiment:
    """Schema for market sentiment data from external sources"""
    source: str  # 'coinmarketcap', 'coingecko', etc.
    coin: str  # 'BTC', 'ETH'
    timestamp: datetime
    
    # Community metrics
    community_score: Optional[float] = None
    social_volume: Optional[int] = None
    social_dominance: Optional[float] = None
    developer_activity: Optional[float] = None
    
    # Sentiment indicators
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[str] = None
    fear_greed_index: Optional[float] = None
    
    # Market data context
    price_usd: Optional[float] = None
    volume_24h: Optional[float] = None
    market_cap: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

@dataclass
class AggregatedSentiment:
    """Schema for daily aggregated sentiment data"""
    date: str  # YYYY-MM-DD format
    coin: str  # 'BTC', 'ETH'
    
    # Reddit sentiment aggregation
    reddit_sentiment_avg: float
    reddit_post_count: int
    reddit_comment_count: int
    reddit_total_score: int
    reddit_bullish_posts: int
    reddit_bearish_posts: int
    reddit_neutral_posts: int
    
    # Market sentiment aggregation
    market_sentiment_avg: Optional[float] = None
    community_score_avg: Optional[float] = None
    social_volume_avg: Optional[float] = None
    
    # Combined metrics
    combined_sentiment: float = 0.0
    confidence_level: float = 0.0
    signal_strength: str = "weak"  # 'weak', 'moderate', 'strong'
    
    # Notable events
    notable_posts: Optional[List[str]] = None  # Post IDs of highly relevant posts
    sentiment_events: Optional[List[str]] = None  # Significant sentiment changes
    
    # Processing metadata
    last_updated: Optional[datetime] = None
    data_sources: Optional[List[str]] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        if self.last_updated:
            data['last_updated'] = self.last_updated.isoformat()
        return data

@dataclass
class CacheMetadata:
    """Schema for cache metadata"""
    cache_key: str
    last_updated: datetime
    data_sources: List[str]
    record_count: int
    expires_at: Optional[datetime] = None
    cache_version: str = "1.0"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['last_updated'] = self.last_updated.isoformat()
        if self.expires_at:
            data['expires_at'] = self.expires_at.isoformat()
        return data
    
    def is_expired(self) -> bool:
        """Check if cache has expired"""
        if not self.expires_at:
            return False
        return datetime.now() > self.expires_at

class SentimentDataEncoder(json.JSONEncoder):
    """Custom JSON encoder for sentiment data objects"""
    def default(self, obj):
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

def save_sentiment_data(data: Union[List, Dict], filepath: str):
    """Save sentiment data to JSON file with proper encoding"""
    with open(filepath, 'w') as f:
        json.dump(data, f, cls=SentimentDataEncoder, indent=2)

def load_sentiment_data(filepath: str) -> Union[List, Dict]:
    """Load sentiment data from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)