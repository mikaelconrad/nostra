#!/usr/bin/env python3
"""
Migration script to move from CSV files to SQLite database
"""

import sys
import os
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from utils.logger import setup_logger
from database.manager import DatabaseManager

logger = setup_logger(__name__)

def main():
    """Main migration function"""
    parser = argparse.ArgumentParser(description='Migrate cryptocurrency data from CSV to database')
    parser.add_argument('--force', action='store_true', help='Force migration even if database exists')
    parser.add_argument('--test', action='store_true', help='Run in test mode (use test database)')
    args = parser.parse_args()
    
    # Set up database URL
    if args.test:
        db_url = 'sqlite:///test_crypto_investment.db'
        logger.info("Running in TEST mode")
    else:
        db_url = config.DATABASE_URL
    
    logger.info("="*60)
    logger.info("Cryptocurrency Investment App - Database Migration")
    logger.info("="*60)
    logger.info(f"Database: {db_url}")
    
    # Check if database already exists
    db_path = db_url.replace('sqlite:///', '')
    if os.path.exists(db_path) and not args.force:
        logger.warning("Database already exists!")
        logger.warning("Use --force to overwrite existing database")
        return 1
    
    # Initialize database
    logger.info("Initializing database...")
    db = DatabaseManager(db_url)
    
    # Run migration
    try:
        logger.info("Starting migration from CSV files...")
        db.migrate_from_csv()
        
        logger.info("="*60)
        logger.info("✅ Migration completed successfully!")
        logger.info("="*60)
        
        # Print summary
        logger.info("\nMigration Summary:")
        
        # Check price data
        for symbol in ['BTC', 'ETH', 'XRP']:
            df = db.get_price_data(symbol, days=1)
            if not df.empty:
                logger.info(f"  {symbol}: {len(df)} price records")
        
        # Check portfolio
        portfolio = db.get_or_create_portfolio()
        logger.info(f"\nPortfolio migrated:")
        logger.info(f"  Cash: CHF {portfolio.cash:.2f}")
        logger.info(f"  Holdings: {len(portfolio.holdings)}")
        
        # Check transactions
        transactions = db.get_transactions(portfolio.id)
        logger.info(f"  Transactions: {len(transactions)}")
        
        logger.info("\n🎉 You can now use the database-backed version!")
        logger.info("Add 'USE_DATABASE=True' to your .env file to enable it.")
        
        return 0
        
    except Exception as e:
        logger.error(f"Migration failed: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())