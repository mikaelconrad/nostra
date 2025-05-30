"""
Database manager for the cryptocurrency investment app
"""

import os
import sys
from sqlalchemy import create_engine, and_, desc, func
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.exc import IntegrityError
from datetime import datetime, timedelta
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils.logger import setup_logger
from utils.error_handlers import DatabaseError, handle_error
from database.models import (
    Base, CryptoPrice, Portfolio, Holding, Transaction, 
    SentimentData, Recommendation, TechnicalIndicator
)

logger = setup_logger(__name__)

class DatabaseManager:
    """Manager class for database operations"""
    
    def __init__(self, database_url=None):
        """Initialize database connection"""
        self.database_url = database_url or config.DATABASE_URL
        
        # Create engine
        self.engine = create_engine(
            self.database_url,
            echo=False,  # Set to True for SQL debugging
            pool_size=5,
            max_overflow=10
        )
        
        # Create session factory
        self.Session = scoped_session(sessionmaker(bind=self.engine))
        
        # Create tables if they don't exist
        self.create_tables()
        
        logger.info(f"Database initialized: {self.database_url}")
    
    def create_tables(self):
        """Create all tables in the database"""
        try:
            Base.metadata.create_all(self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            handle_error(DatabaseError, "Failed to create database tables", e)
    
    def get_session(self):
        """Get a database session"""
        return self.Session()
    
    def close_session(self):
        """Close the current session"""
        self.Session.remove()
    
    # Portfolio operations
    def get_or_create_portfolio(self, user_id='default'):
        """Get or create a portfolio for a user"""
        session = self.get_session()
        try:
            portfolio = session.query(Portfolio).filter_by(user_id=user_id).first()
            
            if not portfolio:
                portfolio = Portfolio(
                    user_id=user_id,
                    cash=config.INITIAL_INVESTMENT
                )
                session.add(portfolio)
                session.commit()
                logger.info(f"Created new portfolio for user: {user_id}")
                
            # Return the portfolio (session will be closed by caller)
            return portfolio
            
        except Exception as e:
            session.rollback()
            handle_error(DatabaseError, "Failed to get/create portfolio", e)
            
    def get_portfolio_id(self, user_id='default'):
        """Get portfolio ID for a user"""
        session = self.get_session()
        try:
            portfolio = session.query(Portfolio).filter_by(user_id=user_id).first()
            if portfolio:
                return portfolio.id
            return None
        finally:
            session.close()
    
    def update_portfolio_cash(self, portfolio_id, amount):
        """Update portfolio cash balance"""
        session = self.get_session()
        try:
            portfolio = session.query(Portfolio).filter_by(id=portfolio_id).first()
            if portfolio:
                portfolio.cash = amount
                session.commit()
                logger.info(f"Updated portfolio cash to: {amount}")
            
        except Exception as e:
            session.rollback()
            handle_error(DatabaseError, "Failed to update portfolio cash", e)
        finally:
            session.close()
    
    def get_or_create_holding(self, portfolio_id, symbol):
        """Get or create a holding for a portfolio"""
        session = self.get_session()
        try:
            holding = session.query(Holding).filter_by(
                portfolio_id=portfolio_id,
                symbol=symbol
            ).first()
            
            if not holding:
                holding = Holding(
                    portfolio_id=portfolio_id,
                    symbol=symbol,
                    amount=0.0
                )
                session.add(holding)
                session.commit()
            
            return holding
            
        except Exception as e:
            session.rollback()
            handle_error(DatabaseError, "Failed to get/create holding", e)
        finally:
            session.close()
    
    def update_holding(self, portfolio_id, symbol, amount):
        """Update holding amount"""
        session = self.get_session()
        try:
            holding = self.get_or_create_holding(portfolio_id, symbol)
            
            # Re-query in this session
            holding = session.query(Holding).filter_by(
                portfolio_id=portfolio_id,
                symbol=symbol
            ).first()
            
            if amount <= 0.00000001:  # Essentially zero
                session.delete(holding)
            else:
                holding.amount = amount
            
            session.commit()
            logger.info(f"Updated holding: {symbol} = {amount}")
            
        except Exception as e:
            session.rollback()
            handle_error(DatabaseError, "Failed to update holding", e)
        finally:
            session.close()
    
    def add_transaction(self, portfolio_id, transaction_data):
        """Add a transaction to the database"""
        session = self.get_session()
        try:
            transaction = Transaction(
                portfolio_id=portfolio_id,
                **transaction_data
            )
            session.add(transaction)
            session.commit()
            logger.info(f"Added transaction: {transaction_data['type']} {transaction_data['amount']} {transaction_data['symbol']}")
            return transaction
            
        except Exception as e:
            session.rollback()
            handle_error(DatabaseError, "Failed to add transaction", e)
        finally:
            session.close()
    
    def get_transactions(self, portfolio_id, limit=None):
        """Get transactions for a portfolio"""
        session = self.get_session()
        try:
            query = session.query(Transaction).filter_by(
                portfolio_id=portfolio_id
            ).order_by(desc(Transaction.timestamp))
            
            if limit:
                query = query.limit(limit)
            
            return [tx.to_dict() for tx in query.all()]
            
        except Exception as e:
            handle_error(DatabaseError, "Failed to get transactions", e)
        finally:
            session.close()
    
    # Price data operations
    def save_price_data(self, symbol, df):
        """Save price data from DataFrame to database"""
        session = self.get_session()
        try:
            records_added = 0
            
            for _, row in df.iterrows():
                # Parse date
                if isinstance(row['date'], str):
                    date = datetime.strptime(row['date'], '%Y-%m-%d')
                else:
                    date = row['date']
                
                # Check if record exists
                existing = session.query(CryptoPrice).filter_by(
                    symbol=symbol,
                    date=date
                ).first()
                
                if not existing:
                    price = CryptoPrice(
                        symbol=symbol,
                        date=date,
                        open=row.get('open'),
                        high=row.get('high'),
                        low=row.get('low'),
                        close=row.get('close'),
                        volume=row.get('volume')
                    )
                    session.add(price)
                    records_added += 1
            
            session.commit()
            logger.info(f"Added {records_added} price records for {symbol}")
            
        except Exception as e:
            session.rollback()
            handle_error(DatabaseError, f"Failed to save price data for {symbol}", e)
        finally:
            session.close()
    
    def get_price_data(self, symbol, days=None, start_date=None, end_date=None):
        """Get price data from database"""
        session = self.get_session()
        try:
            query = session.query(CryptoPrice).filter_by(symbol=symbol)
            
            if days:
                start_date = datetime.utcnow() - timedelta(days=days)
                query = query.filter(CryptoPrice.date >= start_date)
            elif start_date and end_date:
                query = query.filter(
                    and_(CryptoPrice.date >= start_date, 
                         CryptoPrice.date <= end_date)
                )
            
            query = query.order_by(CryptoPrice.date)
            
            # Convert to DataFrame
            data = [price.to_dict() for price in query.all()]
            if data:
                df = pd.DataFrame(data)
                df['date'] = pd.to_datetime(df['date'])
                return df
            else:
                return pd.DataFrame()
            
        except Exception as e:
            handle_error(DatabaseError, f"Failed to get price data for {symbol}", e)
        finally:
            session.close()
    
    def get_latest_price(self, symbol):
        """Get the latest price for a symbol"""
        session = self.get_session()
        try:
            price = session.query(CryptoPrice).filter_by(
                symbol=symbol
            ).order_by(desc(CryptoPrice.date)).first()
            
            return price.close if price else None
            
        except Exception as e:
            handle_error(DatabaseError, f"Failed to get latest price for {symbol}", e)
        finally:
            session.close()
    
    # Sentiment data operations
    def save_sentiment_data(self, symbol, df):
        """Save sentiment data to database"""
        session = self.get_session()
        try:
            records_added = 0
            
            for _, row in df.iterrows():
                # Parse date
                if isinstance(row.get('date'), str):
                    date = datetime.strptime(row['date'], '%Y-%m-%d')
                else:
                    date = row.get('date', datetime.utcnow())
                
                # Check if record exists
                existing = session.query(SentimentData).filter_by(
                    symbol=symbol,
                    date=date
                ).first()
                
                if not existing:
                    sentiment = SentimentData(
                        symbol=symbol,
                        date=date,
                        sentiment_score=row.get('sentiment_score'),
                        vader_compound=row.get('vader_compound'),
                        textblob_polarity=row.get('textblob_polarity'),
                        overall_sentiment=row.get('overall_sentiment'),
                        tweet_count=row.get('tweet_count', 0)
                    )
                    session.add(sentiment)
                    records_added += 1
            
            session.commit()
            logger.info(f"Added {records_added} sentiment records for {symbol}")
            
        except Exception as e:
            session.rollback()
            handle_error(DatabaseError, f"Failed to save sentiment data for {symbol}", e)
        finally:
            session.close()
    
    def get_sentiment_data(self, symbol, days=7):
        """Get sentiment data from database"""
        session = self.get_session()
        try:
            start_date = datetime.utcnow() - timedelta(days=days)
            
            sentiments = session.query(SentimentData).filter(
                and_(SentimentData.symbol == symbol,
                     SentimentData.date >= start_date)
            ).order_by(SentimentData.date).all()
            
            return [s.to_dict() for s in sentiments]
            
        except Exception as e:
            handle_error(DatabaseError, f"Failed to get sentiment data for {symbol}", e)
        finally:
            session.close()
    
    # Recommendation operations
    def save_recommendation(self, recommendation_data):
        """Save a recommendation to database"""
        session = self.get_session()
        try:
            recommendation = Recommendation(**recommendation_data)
            session.add(recommendation)
            session.commit()
            logger.info(f"Saved recommendation for {recommendation_data['symbol']}")
            return recommendation
            
        except Exception as e:
            session.rollback()
            handle_error(DatabaseError, "Failed to save recommendation", e)
        finally:
            session.close()
    
    def get_latest_recommendations(self):
        """Get the latest recommendations for all symbols"""
        session = self.get_session()
        try:
            # Subquery to get latest date for each symbol
            subquery = session.query(
                Recommendation.symbol,
                func.max(Recommendation.date).label('max_date')
            ).group_by(Recommendation.symbol).subquery()
            
            # Join to get full recommendation data
            recommendations = session.query(Recommendation).join(
                subquery,
                and_(Recommendation.symbol == subquery.c.symbol,
                     Recommendation.date == subquery.c.max_date)
            ).all()
            
            return {r.symbol: r.to_dict() for r in recommendations}
            
        except Exception as e:
            handle_error(DatabaseError, "Failed to get latest recommendations", e)
        finally:
            session.close()
    
    # Migration utilities
    def migrate_from_csv(self):
        """Migrate data from CSV files to database"""
        logger.info("Starting CSV to database migration...")
        
        # Migrate price data
        for symbol in ['BTC', 'ETH', 'XRP']:
            csv_path = os.path.join(config.RAW_DATA_DIRECTORY, f'{symbol}_USD.csv')
            if os.path.exists(csv_path):
                logger.info(f"Migrating price data for {symbol}...")
                df = pd.read_csv(csv_path)
                self.save_price_data(symbol, df)
        
        # Migrate sentiment data
        for symbol in ['BTC', 'ETH', 'XRP']:
            csv_path = os.path.join(config.SENTIMENT_DATA_DIRECTORY, f'{symbol}_daily_sentiment.csv')
            if os.path.exists(csv_path):
                logger.info(f"Migrating sentiment data for {symbol}...")
                df = pd.read_csv(csv_path)
                self.save_sentiment_data(symbol, df)
        
        # Migrate portfolio data
        if os.path.exists(config.PORTFOLIO_FILE):
            import json
            logger.info("Migrating portfolio data...")
            with open(config.PORTFOLIO_FILE, 'r') as f:
                portfolio_data = json.load(f)
            
            portfolio = self.get_or_create_portfolio()
            self.update_portfolio_cash(portfolio.id, portfolio_data.get('cash', 0))
            
            for symbol, amount in portfolio_data.get('holdings', {}).items():
                self.update_holding(portfolio.id, symbol, amount)
        
        # Migrate transaction history
        if os.path.exists(config.TRANSACTION_HISTORY_FILE):
            import json
            logger.info("Migrating transaction history...")
            with open(config.TRANSACTION_HISTORY_FILE, 'r') as f:
                transactions = json.load(f)
            
            portfolio = self.get_or_create_portfolio()
            for tx in transactions:
                if isinstance(tx.get('timestamp'), str):
                    tx['timestamp'] = datetime.fromisoformat(tx['timestamp'])
                self.add_transaction(portfolio.id, tx)
        
        logger.info("Migration completed successfully!")

# Global database instance
db = DatabaseManager()

if __name__ == "__main__":
    # Test database connection
    db = DatabaseManager()
    print("Database connected successfully!")
    
    # Optional: Run migration
    # db.migrate_from_csv()