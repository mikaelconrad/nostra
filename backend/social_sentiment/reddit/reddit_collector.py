"""
Reddit API integration for collecting cryptocurrency sentiment data
"""
import praw
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Generator
from dataclasses import asdict
import os
import json

from ..schemas import RedditPost, RedditComment, save_sentiment_data
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from utils.logger import setup_logger

logger = setup_logger(__name__)

class RedditCollector:
    """Collects cryptocurrency-related posts and comments from Reddit"""
    
    def __init__(self, client_id: str = None, client_secret: str = None, user_agent: str = None):
        """
        Initialize Reddit collector
        
        Args:
            client_id: Reddit app client ID (or use REDDIT_CLIENT_ID env var)
            client_secret: Reddit app client secret (or use REDDIT_CLIENT_SECRET env var) 
            user_agent: User agent string (or use REDDIT_USER_AGENT env var)
        """
        # Load environment variables from .env file
        from dotenv import load_dotenv
        load_dotenv()
        
        self.client_id = client_id or os.getenv('REDDIT_CLIENT_ID')
        self.client_secret = client_secret or os.getenv('REDDIT_CLIENT_SECRET')
        self.user_agent = user_agent or os.getenv('REDDIT_USER_AGENT', 'CryptoSentimentCollector/1.0')
        
        # Subreddits to monitor (ordered by priority)
        self.primary_subreddits = ['Bitcoin', 'ethereum', 'CryptoCurrency']
        self.secondary_subreddits = ['btc', 'ethfinance', 'CryptoMarkets']
        
        # Rate limiting settings
        self.requests_per_minute = 60
        self.request_delay = 60 / self.requests_per_minute
        self.last_request_time = 0
        
        # Crypto keywords for relevance scoring
        self.crypto_keywords = {
            'btc': ['bitcoin', 'btc', '$btc', 'satoshi', 'sats'],
            'eth': ['ethereum', 'eth', '$eth', 'ether', 'vitalik'],
            'general': ['crypto', 'cryptocurrency', 'blockchain', 'defi', 'hodl', 'moon', 'diamond hands']
        }
        
        self.reddit = None
        self._initialize_reddit()
    
    def _initialize_reddit(self):
        """Initialize Reddit API connection"""
        try:
            if not all([self.client_id, self.client_secret]):
                logger.warning("Reddit credentials not found. Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET env vars")
                return
                
            self.reddit = praw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                user_agent=self.user_agent,
                ratelimit_seconds=600  # Wait 10 minutes if rate limited
            )
            
            # Test connection
            self.reddit.user.me()
            logger.info("Reddit API connection established")
            
        except Exception as e:
            logger.error(f"Failed to initialize Reddit API: {e}")
            self.reddit = None
    
    def _rate_limit(self):
        """Implement rate limiting for API requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.request_delay:
            sleep_time = self.request_delay - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _calculate_relevance_score(self, text: str, coin: str = None) -> float:
        """
        Calculate relevance score for a post/comment based on crypto keywords
        
        Args:
            text: Post title and content combined
            coin: Specific coin to weight higher ('btc', 'eth')
            
        Returns:
            Relevance score between 0.0 and 1.0
        """
        text_lower = text.lower()
        score = 0.0
        
        # Check for specific coin keywords
        if coin and coin in self.crypto_keywords:
            for keyword in self.crypto_keywords[coin]:
                if keyword in text_lower:
                    score += 0.3
        
        # Check for general crypto keywords
        for keyword in self.crypto_keywords['general']:
            if keyword in text_lower:
                score += 0.1
        
        # Check for other coin keywords
        for coin_key, keywords in self.crypto_keywords.items():
            if coin_key != coin and coin_key != 'general':
                for keyword in keywords:
                    if keyword in text_lower:
                        score += 0.1
        
        return min(score, 1.0)  # Cap at 1.0
    
    def collect_subreddit_posts(self, subreddit_name: str, time_filter: str = 'day', 
                               limit: int = 25, min_score: int = 10) -> List[RedditPost]:
        """
        Collect posts from a specific subreddit
        
        Args:
            subreddit_name: Name of subreddit to collect from
            time_filter: Time filter ('day', 'week', 'month')
            limit: Maximum number of posts to collect
            min_score: Minimum score threshold for posts
            
        Returns:
            List of RedditPost objects
        """
        if not self.reddit:
            logger.error("Reddit API not initialized")
            return []
        
        posts = []
        
        try:
            self._rate_limit()
            subreddit = self.reddit.subreddit(subreddit_name)
            
            logger.info(f"Collecting posts from r/{subreddit_name}")
            
            # Get top posts for the time period
            for submission in subreddit.top(time_filter=time_filter, limit=limit):
                self._rate_limit()
                
                if submission.score < min_score:
                    continue
                
                # Calculate relevance score
                full_text = f"{submission.title} {submission.selftext}"
                relevance_score = self._calculate_relevance_score(full_text)
                
                # Skip posts with very low relevance
                if relevance_score < 0.1:
                    continue
                
                post = RedditPost(
                    id=submission.id,
                    title=submission.title,
                    content=submission.selftext or "",
                    subreddit=subreddit_name,
                    author=str(submission.author) if submission.author else "[deleted]",
                    score=submission.score,
                    upvote_ratio=submission.upvote_ratio,
                    num_comments=submission.num_comments,
                    created_utc=submission.created_utc,
                    url=submission.url,
                    processed_at=datetime.now(),
                    relevance_score=relevance_score
                )
                
                posts.append(post)
                logger.debug(f"Collected post: {post.title[:50]}... (score: {post.score})")
            
            logger.info(f"Collected {len(posts)} posts from r/{subreddit_name}")
            
        except Exception as e:
            logger.error(f"Error collecting posts from r/{subreddit_name}: {e}")
        
        return posts
    
    def collect_post_comments(self, post_id: str, limit: int = 50, 
                             min_score: int = 5) -> List[RedditComment]:
        """
        Collect comments from a specific post
        
        Args:
            post_id: Reddit post ID
            limit: Maximum number of comments to collect
            min_score: Minimum score threshold for comments
            
        Returns:
            List of RedditComment objects
        """
        if not self.reddit:
            logger.error("Reddit API not initialized")
            return []
        
        comments = []
        
        try:
            self._rate_limit()
            submission = self.reddit.submission(id=post_id)
            submission.comments.replace_more(limit=0)  # Don't expand "more comments"
            
            for comment in submission.comments.list()[:limit]:
                if comment.score < min_score:
                    continue
                
                # Calculate relevance score
                relevance_score = self._calculate_relevance_score(comment.body)
                
                # Skip comments with very low relevance
                if relevance_score < 0.1:
                    continue
                
                reddit_comment = RedditComment(
                    id=comment.id,
                    post_id=post_id,
                    body=comment.body,
                    author=str(comment.author) if comment.author else "[deleted]",
                    score=comment.score,
                    created_utc=comment.created_utc,
                    processed_at=datetime.now(),
                    relevance_score=relevance_score
                )
                
                comments.append(reddit_comment)
            
            logger.debug(f"Collected {len(comments)} comments from post {post_id}")
            
        except Exception as e:
            logger.error(f"Error collecting comments from post {post_id}: {e}")
        
        return comments
    
    def collect_daily_data(self, target_date: datetime = None, 
                          subreddit_priority: str = 'primary') -> Dict:
        """
        Collect a day's worth of Reddit data
        
        Args:
            target_date: Date to collect data for (defaults to today)
            subreddit_priority: 'primary', 'secondary', or 'all'
            
        Returns:
            Dictionary with collected posts and comments
        """
        if target_date is None:
            target_date = datetime.now()
        
        # Select subreddits based on priority
        if subreddit_priority == 'primary':
            subreddits = self.primary_subreddits
        elif subreddit_priority == 'secondary':
            subreddits = self.secondary_subreddits
        else:  # 'all'
            subreddits = self.primary_subreddits + self.secondary_subreddits
        
        all_posts = []
        all_comments = []
        
        logger.info(f"Starting daily data collection for {target_date.strftime('%Y-%m-%d')}")
        
        for subreddit in subreddits:
            try:
                # Collect posts
                posts = self.collect_subreddit_posts(subreddit, time_filter='day', limit=25)
                all_posts.extend(posts)
                
                # Collect comments from top posts
                for post in posts[:5]:  # Only get comments from top 5 posts per subreddit
                    comments = self.collect_post_comments(post.id, limit=20)
                    all_comments.extend(comments)
                
                # Add delay between subreddits to be respectful
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Error collecting from r/{subreddit}: {e}")
                continue
        
        data = {
            'date': target_date.strftime('%Y-%m-%d'),
            'posts': [post.to_dict() for post in all_posts],
            'comments': [comment.to_dict() for comment in all_comments],
            'collection_metadata': {
                'collected_at': datetime.now().isoformat(),
                'subreddits': subreddits,
                'total_posts': len(all_posts),
                'total_comments': len(all_comments)
            }
        }
        
        logger.info(f"Daily collection complete: {len(all_posts)} posts, {len(all_comments)} comments")
        
        return data
    
    def save_daily_data(self, data: Dict, base_path: str = "data/social_sentiment/daily"):
        """
        Save daily collection data to file
        
        Args:
            data: Data dictionary from collect_daily_data
            base_path: Base directory for saving files
        """
        date_str = data['date']
        filename = f"reddit_{date_str}.json"
        filepath = os.path.join(base_path, filename)
        
        # Create directory if it doesn't exist
        os.makedirs(base_path, exist_ok=True)
        
        save_sentiment_data(data, filepath)
        logger.info(f"Saved daily Reddit data to {filepath}")
        
        return filepath

# Configuration for Reddit app setup
REDDIT_SETUP_INSTRUCTIONS = """
To use Reddit API integration:

1. Go to https://www.reddit.com/prefs/apps
2. Click "Create App" or "Create Another App"
3. Choose "script" as the app type
4. Fill in the form:
   - Name: CryptoSentimentCollector
   - Description: Collecting cryptocurrency sentiment data
   - About URL: (leave blank)
   - Redirect URI: http://localhost:8080
5. Note down the client ID (under the app name) and client secret
6. Set environment variables:
   export REDDIT_CLIENT_ID="your_client_id"
   export REDDIT_CLIENT_SECRET="your_client_secret"
   export REDDIT_USER_AGENT="CryptoSentimentCollector/1.0"
"""