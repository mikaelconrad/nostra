"""
Run the sentiment data collection service
"""
import sys
import os
import time
import signal
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.social_sentiment.data_collection_service import (
    start_sentiment_service, stop_sentiment_service, get_sentiment_service
)
from utils.logger import setup_logger

logger = setup_logger(__name__)

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info("Received shutdown signal, stopping sentiment service...")
    stop_sentiment_service()
    sys.exit(0)

def main():
    """Main function to run the sentiment service"""
    print("=" * 60)
    print("🎭 Starting Crypto Sentiment Data Collection Service")
    print("=" * 60)
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Service configuration
    config = {
        'reddit_collection_interval': 'daily',  # Collect 3 times per day
        'market_collection_interval': 'daily',  # Collect 2 times per day
        'max_retries': 3,
        'retry_delay': 300,  # 5 minutes
        'primary_subreddits_only': True,  # Focus on main subreddits
        'debug_mode': False
    }
    
    try:
        # Start the service
        service = start_sentiment_service(config)
        
        print(f"✅ Service started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n📋 Service Configuration:")
        print(f"   Reddit Collection: {config['reddit_collection_interval']}")
        print(f"   Market Collection: {config['market_collection_interval']}")
        print(f"   Max Retries: {config['max_retries']}")
        print(f"   Primary Subreddits Only: {config['primary_subreddits_only']}")
        
        print("\n⏰ Scheduled Collections:")
        print("   Reddit Data: 09:00, 15:00, 21:00 daily")
        print("   Market Data: 08:30, 16:30 daily")
        print("   Daily Aggregation: 23:30 daily")
        print("   Weekly Cleanup: 02:00 Sunday")
        
        print("\n🔧 Available Commands:")
        print("   Ctrl+C - Stop service")
        print("   Check status - View service status")
        print("   Collect now - Trigger immediate collection")
        
        print("\n📊 Service Status:")
        
        # Main service loop
        while True:
            try:
                # Show status every 5 minutes
                status = service.get_service_status()
                cache_stats = status['cache_stats']
                
                print(f"\n⏰ {datetime.now().strftime('%H:%M:%S')} - Service Running")
                print(f"   📊 Cache: {cache_stats['total_files']} files, {cache_stats['total_size_mb']:.1f} MB")
                print(f"   📈 Collections: Reddit={status['collection_stats']['reddit_collections']}, Market={status['collection_stats']['market_collections']}")
                print(f"   ❌ Failed: {status['collection_stats']['failed_collections']}")
                
                if status['collection_stats']['last_successful_collection']:
                    last_success = status['collection_stats']['last_successful_collection']
                    print(f"   ✅ Last Success: {last_success}")
                
                if status['collection_stats']['last_error']:
                    print(f"   ⚠️  Last Error: {status['collection_stats']['last_error']}")
                
                # Check health
                health = service.get_collection_health()
                if health['status'] != 'healthy':
                    print(f"   🏥 Health: {health['status'].upper()}")
                    for issue in health['issues']:
                        print(f"      - {issue}")
                
                time.sleep(300)  # Wait 5 minutes
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Service loop error: {e}")
                time.sleep(60)  # Wait 1 minute on error
    
    except Exception as e:
        logger.error(f"Failed to start sentiment service: {e}")
        print(f"❌ Failed to start service: {e}")
        return 1
    
    finally:
        print("\n🛑 Stopping sentiment service...")
        stop_sentiment_service()
        print("✅ Service stopped")
    
    return 0

def run_one_time_collection():
    """Run a one-time collection for testing"""
    print("🎭 Running One-Time Sentiment Data Collection")
    print("=" * 50)
    
    try:
        # Create service without starting scheduler
        service = get_sentiment_service({
            'reddit_collection_interval': 'manual',
            'market_collection_interval': 'manual',
            'max_retries': 2,
            'retry_delay': 30,
            'primary_subreddits_only': True,
            'debug_mode': True
        })
        
        print("📊 Starting data collection...")
        
        # Collect all data types
        results = service.collect_now('all', force=True)
        
        print("\n✅ Collection Results:")
        for data_type, result in results.items():
            if result:
                print(f"   {data_type.title()}: {result}")
        
        # Show final status
        status = service.get_service_status()
        cache_stats = status['cache_stats']
        
        print(f"\n📈 Final Statistics:")
        print(f"   Cache Files: {cache_stats['total_files']}")
        print(f"   Cache Size: {cache_stats['total_size_mb']:.1f} MB")
        print(f"   Available Dates: {len(cache_stats['available_dates'])}")
        print(f"   Data Types: {', '.join(cache_stats['data_types'])}")
        
        print("\n🎉 One-time collection completed!")
        
    except Exception as e:
        logger.error(f"One-time collection failed: {e}")
        print(f"❌ Collection failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "once":
        exit_code = run_one_time_collection()
    else:
        exit_code = main()
    
    sys.exit(exit_code)