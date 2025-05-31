"""
Caching and data management for social sentiment analysis
"""
import os
import json
import gzip
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import logging
from pathlib import Path

from .schemas import CacheMetadata, save_sentiment_data, load_sentiment_data
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.logger import setup_logger

logger = setup_logger(__name__)

class SentimentCacheManager:
    """Manages caching and data storage for sentiment analysis"""
    
    def __init__(self, base_cache_dir: str = "data/social_sentiment"):
        """
        Initialize cache manager
        
        Args:
            base_cache_dir: Base directory for all sentiment data
        """
        self.base_cache_dir = Path(base_cache_dir)
        self.daily_dir = self.base_cache_dir / "daily"
        self.cache_dir = self.base_cache_dir / "cache"
        self.aggregated_dir = self.base_cache_dir / "aggregated"
        
        # Create directories
        for directory in [self.daily_dir, self.cache_dir, self.aggregated_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Cache settings
        self.cache_expiry_hours = 24  # Default cache expiry
        self.compression_age_days = 30  # Compress data older than 30 days
        self.retention_days = 365  # Keep data for 1 year
        
        logger.info(f"Initialized sentiment cache manager at {self.base_cache_dir}")
    
    def _get_cache_key(self, data_type: str, date: str, source: str = None) -> str:
        """
        Generate cache key for data
        
        Args:
            data_type: Type of data ('reddit', 'market', 'aggregated')
            date: Date string (YYYY-MM-DD)
            source: Optional source identifier
            
        Returns:
            Cache key string
        """
        if source:
            return f"{data_type}_{source}_{date}"
        return f"{data_type}_{date}"
    
    def _get_cache_metadata_path(self, cache_key: str) -> Path:
        """Get path for cache metadata file"""
        return self.cache_dir / f"{cache_key}_metadata.json"
    
    def _get_daily_data_path(self, data_type: str, date: str, source: str = None) -> Path:
        """Get path for daily data file"""
        if source:
            filename = f"{data_type}_{source}_{date}.json"
        else:
            filename = f"{data_type}_{date}.json"
        return self.daily_dir / filename
    
    def _get_compressed_data_path(self, data_type: str, date: str, source: str = None) -> Path:
        """Get path for compressed data file"""
        if source:
            filename = f"{data_type}_{source}_{date}.json.gz"
        else:
            filename = f"{data_type}_{date}.json.gz"
        return self.daily_dir / filename
    
    def _get_aggregated_data_path(self, date: str, coin: str = None) -> Path:
        """Get path for aggregated data file"""
        if coin:
            filename = f"aggregated_{coin}_{date}.json"
        else:
            filename = f"aggregated_{date}.json"
        return self.aggregated_dir / filename
    
    def save_cache_metadata(self, cache_key: str, metadata: CacheMetadata):
        """Save cache metadata"""
        metadata_path = self._get_cache_metadata_path(cache_key)
        save_sentiment_data(metadata.to_dict(), str(metadata_path))
        logger.debug(f"Saved cache metadata for {cache_key}")
    
    def load_cache_metadata(self, cache_key: str) -> Optional[CacheMetadata]:
        """Load cache metadata"""
        metadata_path = self._get_cache_metadata_path(cache_key)
        
        if not metadata_path.exists():
            return None
        
        try:
            data = load_sentiment_data(str(metadata_path))
            
            # Convert datetime strings back to datetime objects
            if data.get('last_updated'):
                data['last_updated'] = datetime.fromisoformat(data['last_updated'])
            if data.get('expires_at'):
                data['expires_at'] = datetime.fromisoformat(data['expires_at'])
            
            return CacheMetadata(**data)
        except Exception as e:
            logger.error(f"Error loading cache metadata for {cache_key}: {e}")
            return None
    
    def is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        metadata = self.load_cache_metadata(cache_key)
        
        if not metadata:
            return False
        
        return not metadata.is_expired()
    
    def save_daily_data(self, data_type: str, date: str, data: Dict, 
                       source: str = None, data_sources: List[str] = None) -> str:
        """
        Save daily data with automatic caching
        
        Args:
            data_type: Type of data ('reddit', 'market', 'aggregated')
            date: Date string (YYYY-MM-DD)
            data: Data to save
            source: Optional source identifier
            data_sources: List of data sources used
            
        Returns:
            File path where data was saved
        """
        # Save the actual data
        data_path = self._get_daily_data_path(data_type, date, source)
        save_sentiment_data(data, str(data_path))
        
        # Create and save cache metadata
        cache_key = self._get_cache_key(data_type, date, source)
        
        metadata = CacheMetadata(
            cache_key=cache_key,
            last_updated=datetime.now(),
            data_sources=data_sources or [data_type],
            record_count=self._count_records(data),
            expires_at=datetime.now() + timedelta(hours=self.cache_expiry_hours)
        )
        
        self.save_cache_metadata(cache_key, metadata)
        
        logger.info(f"Saved daily {data_type} data for {date} to {data_path}")
        return str(data_path)
    
    def load_daily_data(self, data_type: str, date: str, source: str = None) -> Optional[Dict]:
        """
        Load daily data with cache checking
        
        Args:
            data_type: Type of data ('reddit', 'market', 'aggregated')
            date: Date string (YYYY-MM-DD)
            source: Optional source identifier
            
        Returns:
            Loaded data or None if not found/expired
        """
        cache_key = self._get_cache_key(data_type, date, source)
        
        # Check if cache is valid
        if not self.is_cache_valid(cache_key):
            logger.debug(f"Cache invalid or expired for {cache_key}")
            return None
        
        # Try to load uncompressed data first
        data_path = self._get_daily_data_path(data_type, date, source)
        
        if data_path.exists():
            try:
                return load_sentiment_data(str(data_path))
            except Exception as e:
                logger.error(f"Error loading data from {data_path}: {e}")
        
        # Try to load compressed data
        compressed_path = self._get_compressed_data_path(data_type, date, source)
        
        if compressed_path.exists():
            try:
                return self._load_compressed_data(str(compressed_path))
            except Exception as e:
                logger.error(f"Error loading compressed data from {compressed_path}: {e}")
        
        logger.debug(f"No data found for {cache_key}")
        return None
    
    def _count_records(self, data: Any) -> int:
        """Count records in data structure"""
        if isinstance(data, dict):
            if 'posts' in data and 'comments' in data:
                return len(data['posts']) + len(data['comments'])
            elif 'market_sentiment' in data:
                return len(data['market_sentiment'])
            elif isinstance(data.get('data'), list):
                return len(data['data'])
            else:
                return 1
        elif isinstance(data, list):
            return len(data)
        else:
            return 1
    
    def _load_compressed_data(self, filepath: str) -> Dict:
        """Load data from compressed file"""
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            return json.load(f)
    
    def _save_compressed_data(self, data: Dict, filepath: str):
        """Save data to compressed file"""
        with gzip.open(filepath, 'wt', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    def compress_old_data(self, age_days: int = None) -> int:
        """
        Compress data files older than specified age
        
        Args:
            age_days: Age threshold in days (defaults to self.compression_age_days)
            
        Returns:
            Number of files compressed
        """
        if age_days is None:
            age_days = self.compression_age_days
        
        cutoff_date = datetime.now() - timedelta(days=age_days)
        compressed_count = 0
        
        logger.info(f"Compressing data files older than {age_days} days")
        
        for file_path in self.daily_dir.glob("*.json"):
            try:
                # Check file modification time
                file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                
                if file_mtime < cutoff_date:
                    # Load, compress, and save
                    data = load_sentiment_data(str(file_path))
                    compressed_path = file_path.with_suffix('.json.gz')
                    
                    self._save_compressed_data(data, str(compressed_path))
                    
                    # Remove original file
                    file_path.unlink()
                    
                    compressed_count += 1
                    logger.debug(f"Compressed {file_path.name}")
            
            except Exception as e:
                logger.error(f"Error compressing {file_path}: {e}")
        
        logger.info(f"Compressed {compressed_count} files")
        return compressed_count
    
    def cleanup_old_data(self, retention_days: int = None) -> int:
        """
        Remove data files older than retention period
        
        Args:
            retention_days: Retention period in days (defaults to self.retention_days)
            
        Returns:
            Number of files removed
        """
        if retention_days is None:
            retention_days = self.retention_days
        
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        removed_count = 0
        
        logger.info(f"Removing data files older than {retention_days} days")
        
        # Remove old data files
        for file_path in self.daily_dir.glob("*"):
            try:
                file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                
                if file_mtime < cutoff_date:
                    file_path.unlink()
                    removed_count += 1
                    logger.debug(f"Removed {file_path.name}")
            
            except Exception as e:
                logger.error(f"Error removing {file_path}: {e}")
        
        # Remove old cache metadata
        for file_path in self.cache_dir.glob("*_metadata.json"):
            try:
                file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                
                if file_mtime < cutoff_date:
                    file_path.unlink()
                    logger.debug(f"Removed cache metadata {file_path.name}")
            
            except Exception as e:
                logger.error(f"Error removing cache metadata {file_path}: {e}")
        
        logger.info(f"Removed {removed_count} old files")
        return removed_count
    
    def get_available_dates(self, data_type: str = None, source: str = None) -> List[str]:
        """
        Get list of available dates for data
        
        Args:
            data_type: Filter by data type
            source: Filter by source
            
        Returns:
            List of date strings (YYYY-MM-DD)
        """
        dates = set()
        
        # Search pattern
        if data_type and source:
            patterns = [f"{data_type}_{source}_*.json", f"{data_type}_{source}_*.json.gz"]
        elif data_type:
            patterns = [f"{data_type}_*.json", f"{data_type}_*.json.gz"]
        else:
            patterns = ["*.json", "*.json.gz"]
        
        for pattern in patterns:
            for file_path in self.daily_dir.glob(pattern):
                # Extract date from filename
                filename = file_path.stem
                if filename.endswith('.json'):
                    filename = filename[:-5]  # Remove .json
                
                # Find date pattern (YYYY-MM-DD)
                parts = filename.split('_')
                for part in parts:
                    if len(part) == 10 and part.count('-') == 2:
                        try:
                            datetime.strptime(part, '%Y-%m-%d')
                            dates.add(part)
                        except ValueError:
                            continue
        
        return sorted(list(dates))
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        stats = {
            'total_files': 0,
            'compressed_files': 0,
            'uncompressed_files': 0,
            'total_size_mb': 0,
            'available_dates': [],
            'data_types': set(),
            'oldest_date': None,
            'newest_date': None
        }
        
        for file_path in self.daily_dir.iterdir():
            if file_path.is_file():
                stats['total_files'] += 1
                stats['total_size_mb'] += file_path.stat().st_size / (1024 * 1024)
                
                if file_path.suffix == '.gz':
                    stats['compressed_files'] += 1
                elif file_path.suffix == '.json':
                    stats['uncompressed_files'] += 1
                
                # Extract data type and date
                filename = file_path.stem
                if filename.endswith('.json'):
                    filename = filename[:-5]
                
                parts = filename.split('_')
                if len(parts) >= 2:
                    stats['data_types'].add(parts[0])
                
                # Find date
                for part in parts:
                    if len(part) == 10 and part.count('-') == 2:
                        try:
                            datetime.strptime(part, '%Y-%m-%d')
                            stats['available_dates'].append(part)
                        except ValueError:
                            continue
        
        # Convert set to list and sort dates
        stats['data_types'] = list(stats['data_types'])
        stats['available_dates'] = sorted(list(set(stats['available_dates'])))
        
        if stats['available_dates']:
            stats['oldest_date'] = stats['available_dates'][0]
            stats['newest_date'] = stats['available_dates'][-1]
        
        stats['total_size_mb'] = round(stats['total_size_mb'], 2)
        
        return stats
    
    def force_cache_refresh(self, cache_key: str):
        """Force refresh of cache by removing metadata"""
        metadata_path = self._get_cache_metadata_path(cache_key)
        if metadata_path.exists():
            metadata_path.unlink()
            logger.info(f"Forced cache refresh for {cache_key}")
    
    def warm_cache_for_period(self, start_date: str, end_date: str, 
                             data_types: List[str] = None) -> Dict:
        """
        Warm cache for a specific period (placeholder for future implementation)
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            data_types: List of data types to warm
            
        Returns:
            Status dictionary
        """
        if data_types is None:
            data_types = ['reddit', 'market']
        
        logger.info(f"Cache warming requested for {start_date} to {end_date}")
        
        # This would trigger data collection for missing dates
        # Implementation depends on the collection services
        
        return {
            'status': 'warming_requested',
            'start_date': start_date,
            'end_date': end_date,
            'data_types': data_types
        }