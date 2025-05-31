"""
Background data collection service for social sentiment analysis
"""
import time
import schedule
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from .reddit.reddit_collector import RedditCollector
from .market_data.market_collector import MarketDataCollector
from .sentiment_analyzer import CryptoSentimentAnalyzer
from .cache_manager import SentimentCacheManager
from .schemas import save_sentiment_data
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.logger import setup_logger

logger = setup_logger(__name__)

class SentimentDataCollectionService:
    """Background service for collecting and processing sentiment data"""
    
    def __init__(self, config: Dict = None):
        """
        Initialize the data collection service
        
        Args:
            config: Configuration dictionary with collection settings
        """
        self.config = config or {
            'reddit_collection_interval': 'daily',  # 'hourly', 'daily', 'manual'
            'market_collection_interval': 'daily',
            'max_retries': 3,
            'retry_delay': 300,  # 5 minutes
            'collection_timeout': 1800,  # 30 minutes
            'primary_subreddits_only': True,
            'debug_mode': False
        }
        
        # Initialize components
        self.reddit_collector = RedditCollector()
        self.market_collector = MarketDataCollector()
        self.sentiment_analyzer = CryptoSentimentAnalyzer()
        self.cache_manager = SentimentCacheManager()
        
        # Service state
        self.is_running = False
        self.collection_thread = None
        self.last_reddit_collection = None
        self.last_market_collection = None
        self.collection_stats = {
            'reddit_collections': 0,
            'market_collections': 0,
            'failed_collections': 0,
            'last_successful_collection': None,
            'last_error': None
        }
        
        logger.info("Sentiment data collection service initialized")
    
    def start_service(self):
        """Start the background data collection service"""
        if self.is_running:
            logger.warning("Service is already running")
            return
        
        self.is_running = True
        
        # Schedule collection jobs
        self._schedule_collections()
        
        # Start the scheduler thread
        self.collection_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.collection_thread.start()
        
        logger.info("Sentiment data collection service started")
    
    def stop_service(self):
        """Stop the background data collection service"""
        self.is_running = False
        schedule.clear()
        
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=5)
        
        logger.info("Sentiment data collection service stopped")
    
    def _schedule_collections(self):
        """Schedule collection jobs based on configuration"""
        # Reddit collection scheduling
        if self.config['reddit_collection_interval'] == 'hourly':
            schedule.every().hour.at(":15").do(self._collect_reddit_data_job)
        elif self.config['reddit_collection_interval'] == 'daily':
            schedule.every().day.at("09:00").do(self._collect_reddit_data_job)
            schedule.every().day.at("15:00").do(self._collect_reddit_data_job)
            schedule.every().day.at("21:00").do(self._collect_reddit_data_job)
        
        # Market data collection scheduling
        if self.config['market_collection_interval'] == 'hourly':
            schedule.every().hour.at(":30").do(self._collect_market_data_job)
        elif self.config['market_collection_interval'] == 'daily':
            schedule.every().day.at("08:30").do(self._collect_market_data_job)
            schedule.every().day.at("16:30").do(self._collect_market_data_job)
        
        # Daily aggregation and cleanup
        schedule.every().day.at("23:30").do(self._daily_aggregation_job)
        schedule.every().sunday.at("02:00").do(self._weekly_cleanup_job)
        
        logger.info("Collection jobs scheduled")
    
    def _run_scheduler(self):
        """Run the scheduler loop"""
        logger.info("Scheduler started")
        
        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(300)  # Wait 5 minutes on error
        
        logger.info("Scheduler stopped")
    
    def _collect_reddit_data_job(self):
        """Job function for Reddit data collection"""
        try:
            logger.info("Starting Reddit data collection job")
            
            # Check if we already collected today
            today = datetime.now().strftime('%Y-%m-%d')
            existing_data = self.cache_manager.load_daily_data('reddit', today)
            
            if existing_data and not self.config.get('force_recollection', False):
                logger.info(f"Reddit data already exists for {today}, skipping collection")
                return
            
            # Determine subreddit priority
            priority = 'primary' if self.config['primary_subreddits_only'] else 'all'
            
            # Collect data with retry logic
            success = False
            for attempt in range(self.config['max_retries']):
                try:
                    reddit_data = self.reddit_collector.collect_daily_data(
                        target_date=datetime.now(),
                        subreddit_priority=priority
                    )
                    
                    if reddit_data and reddit_data.get('posts'):
                        # Save to cache
                        self.cache_manager.save_daily_data(
                            'reddit', today, reddit_data,
                            data_sources=['reddit']
                        )
                        
                        self.last_reddit_collection = datetime.now()
                        self.collection_stats['reddit_collections'] += 1
                        self.collection_stats['last_successful_collection'] = datetime.now()
                        
                        logger.info(f"Reddit data collection successful: {len(reddit_data['posts'])} posts, {len(reddit_data['comments'])} comments")
                        success = True
                        break
                    else:
                        raise Exception("No Reddit data collected")
                
                except Exception as e:
                    logger.warning(f"Reddit collection attempt {attempt + 1} failed: {e}")
                    if attempt < self.config['max_retries'] - 1:
                        time.sleep(self.config['retry_delay'])
            
            if not success:
                self.collection_stats['failed_collections'] += 1
                self.collection_stats['last_error'] = f"Reddit collection failed after {self.config['max_retries']} attempts"
                logger.error(self.collection_stats['last_error'])
        
        except Exception as e:
            logger.error(f"Reddit data collection job error: {e}")
            self.collection_stats['failed_collections'] += 1
            self.collection_stats['last_error'] = str(e)
    
    def _collect_market_data_job(self):
        """Job function for market data collection"""
        try:
            logger.info("Starting market data collection job")
            
            # Check if we already collected today
            today = datetime.now().strftime('%Y-%m-%d')
            existing_data = self.cache_manager.load_daily_data('market', today)
            
            if existing_data and not self.config.get('force_recollection', False):
                logger.info(f"Market data already exists for {today}, skipping collection")
                return
            
            # Collect data with retry logic
            success = False
            for attempt in range(self.config['max_retries']):
                try:
                    market_data = self.market_collector.collect_daily_market_sentiment(['BTC', 'ETH'])
                    
                    if market_data and market_data.get('market_sentiment'):
                        # Save to cache
                        self.cache_manager.save_daily_data(
                            'market', today, market_data,
                            data_sources=['coingecko', 'alternative']
                        )
                        
                        self.last_market_collection = datetime.now()
                        self.collection_stats['market_collections'] += 1
                        self.collection_stats['last_successful_collection'] = datetime.now()
                        
                        logger.info(f"Market data collection successful: {len(market_data['market_sentiment'])} records")
                        success = True
                        break
                    else:
                        raise Exception("No market data collected")
                
                except Exception as e:
                    logger.warning(f"Market collection attempt {attempt + 1} failed: {e}")
                    if attempt < self.config['max_retries'] - 1:
                        time.sleep(self.config['retry_delay'])
            
            if not success:
                self.collection_stats['failed_collections'] += 1
                self.collection_stats['last_error'] = f"Market collection failed after {self.config['max_retries']} attempts"
                logger.error(self.collection_stats['last_error'])
        
        except Exception as e:
            logger.error(f"Market data collection job error: {e}")
            self.collection_stats['failed_collections'] += 1
            self.collection_stats['last_error'] = str(e)
    
    def _daily_aggregation_job(self):
        """Job function for daily sentiment aggregation"""
        try:
            logger.info("Starting daily aggregation job")
            
            today = datetime.now().strftime('%Y-%m-%d')
            
            for coin in ['BTC', 'ETH']:
                try:
                    # Check if aggregation already exists
                    existing_data = self.cache_manager.load_daily_data('aggregated', today, coin)
                    if existing_data and not self.config.get('force_reaggregation', False):
                        logger.info(f"Aggregated data already exists for {coin} on {today}")
                        continue
                    
                    # Load Reddit and market data
                    reddit_data = self.cache_manager.load_daily_data('reddit', today)
                    market_data = self.cache_manager.load_daily_data('market', today)
                    
                    if not reddit_data:
                        logger.warning(f"No Reddit data available for aggregation on {today}")
                        continue
                    
                    # Convert to objects and analyze
                    from .schemas import RedditPost, RedditComment
                    
                    posts = [RedditPost(**post) for post in reddit_data.get('posts', [])]
                    comments = [RedditComment(**comment) for comment in reddit_data.get('comments', [])]
                    
                    # Filter posts/comments relevant to the coin
                    coin_keywords = {
                        'BTC': ['bitcoin', 'btc', '$btc', 'satoshi'],
                        'ETH': ['ethereum', 'eth', '$eth', 'ether', 'vitalik']
                    }
                    
                    relevant_posts = []
                    for post in posts:
                        text = f"{post.title} {post.content}".lower()
                        if any(keyword in text for keyword in coin_keywords.get(coin, [])):
                            relevant_posts.append(post)
                    
                    relevant_comments = []
                    for comment in comments:
                        text = comment.body.lower()
                        if any(keyword in text for keyword in coin_keywords.get(coin, [])):
                            relevant_comments.append(comment)
                    
                    if not relevant_posts and not relevant_comments:
                        logger.warning(f"No relevant posts/comments found for {coin} on {today}")
                        continue
                    
                    # Aggregate sentiment
                    aggregated = self.sentiment_analyzer.aggregate_daily_sentiment(
                        relevant_posts, relevant_comments, coin, today
                    )
                    
                    # Add market sentiment if available
                    if market_data and market_data.get('market_sentiment'):
                        market_sentiments = [
                            ms.get('sentiment_score', 0) 
                            for ms in market_data['market_sentiment'] 
                            if ms.get('coin') == coin and ms.get('sentiment_score') is not None
                        ]
                        
                        if market_sentiments:
                            aggregated.market_sentiment_avg = sum(market_sentiments) / len(market_sentiments)
                            
                            # Recalculate combined sentiment
                            reddit_weight = 0.7
                            market_weight = 0.3
                            aggregated.combined_sentiment = (
                                reddit_weight * aggregated.reddit_sentiment_avg + 
                                market_weight * aggregated.market_sentiment_avg
                            )
                            aggregated.data_sources.append('market')
                    
                    # Save aggregated data
                    aggregated_dict = aggregated.to_dict()
                    self.cache_manager.save_daily_data(
                        'aggregated', today, aggregated_dict, coin,
                        data_sources=aggregated.data_sources
                    )
                    
                    logger.info(f"Daily aggregation completed for {coin}: sentiment={aggregated.combined_sentiment:.3f}")
                
                except Exception as e:
                    logger.error(f"Error aggregating data for {coin}: {e}")
        
        except Exception as e:
            logger.error(f"Daily aggregation job error: {e}")
    
    def _weekly_cleanup_job(self):
        """Job function for weekly cache cleanup"""
        try:
            logger.info("Starting weekly cleanup job")
            
            # Compress old data (older than 30 days)
            compressed_count = self.cache_manager.compress_old_data(30)
            
            # Remove very old data (older than 365 days)
            removed_count = self.cache_manager.cleanup_old_data(365)
            
            logger.info(f"Weekly cleanup completed: {compressed_count} files compressed, {removed_count} files removed")
        
        except Exception as e:
            logger.error(f"Weekly cleanup job error: {e}")
    
    def collect_now(self, data_type: str = 'all', force: bool = False) -> Dict:
        """
        Manually trigger data collection
        
        Args:
            data_type: Type of data to collect ('reddit', 'market', 'all')
            force: Force collection even if data already exists
            
        Returns:
            Collection results
        """
        results = {'reddit': None, 'market': None, 'aggregated': None}
        
        try:
            if data_type in ['reddit', 'all']:
                # Temporarily set force recollection
                original_force = self.config.get('force_recollection', False)
                self.config['force_recollection'] = force
                
                self._collect_reddit_data_job()
                results['reddit'] = 'completed'
                
                self.config['force_recollection'] = original_force
            
            if data_type in ['market', 'all']:
                # Temporarily set force recollection
                original_force = self.config.get('force_recollection', False)
                self.config['force_recollection'] = force
                
                self._collect_market_data_job()
                results['market'] = 'completed'
                
                self.config['force_recollection'] = original_force
            
            if data_type in ['all']:
                # Trigger aggregation
                original_force = self.config.get('force_reaggregation', False)
                self.config['force_reaggregation'] = force
                
                self._daily_aggregation_job()
                results['aggregated'] = 'completed'
                
                self.config['force_reaggregation'] = original_force
            
            return results
        
        except Exception as e:
            logger.error(f"Manual collection error: {e}")
            results['error'] = str(e)
            return results
    
    def get_service_status(self) -> Dict:
        """Get current service status and statistics"""
        return {
            'is_running': self.is_running,
            'last_reddit_collection': self.last_reddit_collection.isoformat() if self.last_reddit_collection else None,
            'last_market_collection': self.last_market_collection.isoformat() if self.last_market_collection else None,
            'collection_stats': self.collection_stats.copy(),
            'scheduled_jobs': len(schedule.jobs),
            'cache_stats': self.cache_manager.get_cache_stats(),
            'config': self.config.copy()
        }
    
    def update_config(self, new_config: Dict):
        """Update service configuration"""
        self.config.update(new_config)
        
        # Reschedule if service is running
        if self.is_running:
            schedule.clear()
            self._schedule_collections()
            logger.info("Service configuration updated and jobs rescheduled")
    
    def get_collection_health(self) -> Dict:
        """Check collection health and identify issues"""
        health = {
            'status': 'healthy',
            'issues': [],
            'recommendations': []
        }
        
        # Check if service is running
        if not self.is_running:
            health['status'] = 'stopped'
            health['issues'].append('Service is not running')
            health['recommendations'].append('Start the service')
        
        # Check recent collections
        now = datetime.now()
        if self.last_reddit_collection:
            time_since_reddit = (now - self.last_reddit_collection).total_seconds() / 3600
            if time_since_reddit > 25:  # More than 25 hours
                health['status'] = 'warning'
                health['issues'].append(f'Reddit data not collected for {time_since_reddit:.1f} hours')
                health['recommendations'].append('Check Reddit API credentials and rate limits')
        
        if self.last_market_collection:
            time_since_market = (now - self.last_market_collection).total_seconds() / 3600
            if time_since_market > 25:  # More than 25 hours
                health['status'] = 'warning'
                health['issues'].append(f'Market data not collected for {time_since_market:.1f} hours')
                health['recommendations'].append('Check market data API endpoints')
        
        # Check error rate
        total_collections = (self.collection_stats['reddit_collections'] + 
                           self.collection_stats['market_collections'])
        if total_collections > 0:
            error_rate = self.collection_stats['failed_collections'] / total_collections
            if error_rate > 0.2:  # More than 20% failure rate
                health['status'] = 'warning'
                health['issues'].append(f'High error rate: {error_rate:.1%}')
                health['recommendations'].append('Check logs for recurring errors')
        
        return health


# Global service instance
sentiment_service = None

def get_sentiment_service(config: Dict = None) -> SentimentDataCollectionService:
    """Get or create the global sentiment service instance"""
    global sentiment_service
    
    if sentiment_service is None:
        sentiment_service = SentimentDataCollectionService(config)
    
    return sentiment_service

def start_sentiment_service(config: Dict = None):
    """Start the global sentiment service"""
    service = get_sentiment_service(config)
    service.start_service()
    return service

def stop_sentiment_service():
    """Stop the global sentiment service"""
    global sentiment_service
    
    if sentiment_service:
        sentiment_service.stop_service()
        sentiment_service = None