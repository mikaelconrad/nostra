"""
Database models for the cryptocurrency investment app
"""

from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime
import json

Base = declarative_base()

class CryptoPrice(Base):
    """Model for cryptocurrency price data"""
    __tablename__ = 'crypto_prices'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False, index=True)
    date = Column(DateTime, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float, nullable=False)
    volume = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Composite index for efficient queries
    __table_args__ = (
        Index('idx_symbol_date', 'symbol', 'date', unique=True),
    )
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'symbol': self.symbol,
            'date': self.date.isoformat() if self.date else None,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume
        }

class Portfolio(Base):
    """Model for portfolio data"""
    __tablename__ = 'portfolios'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String(50), default='default', index=True)
    cash = Column(Float, nullable=False, default=0.0)
    last_update = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    holdings = relationship('Holding', back_populates='portfolio', cascade='all, delete-orphan')
    transactions = relationship('Transaction', back_populates='portfolio', cascade='all, delete-orphan')
    
    def to_dict(self):
        """Convert to dictionary"""
        holdings_dict = {}
        for holding in self.holdings:
            holdings_dict[holding.symbol] = holding.amount
            
        return {
            'cash': self.cash,
            'holdings': holdings_dict,
            'last_update': self.last_update.isoformat() if self.last_update else None
        }

class Holding(Base):
    """Model for cryptocurrency holdings"""
    __tablename__ = 'holdings'
    
    id = Column(Integer, primary_key=True)
    portfolio_id = Column(Integer, ForeignKey('portfolios.id'), nullable=False)
    symbol = Column(String(10), nullable=False, index=True)
    amount = Column(Float, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    portfolio = relationship('Portfolio', back_populates='holdings')
    
    # Unique constraint
    __table_args__ = (
        Index('idx_portfolio_symbol', 'portfolio_id', 'symbol', unique=True),
    )

class Transaction(Base):
    """Model for transaction history"""
    __tablename__ = 'transactions'
    
    id = Column(Integer, primary_key=True)
    portfolio_id = Column(Integer, ForeignKey('portfolios.id'), nullable=False)
    type = Column(String(10), nullable=False)  # 'buy' or 'sell'
    symbol = Column(String(10), nullable=False, index=True)
    amount = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    total = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Relationships
    portfolio = relationship('Portfolio', back_populates='transactions')
    
    # Index for efficient queries
    __table_args__ = (
        Index('idx_portfolio_timestamp', 'portfolio_id', 'timestamp'),
    )
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'type': self.type,
            'symbol': self.symbol,
            'amount': self.amount,
            'price': self.price,
            'total': self.total,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }

class SentimentData(Base):
    """Model for sentiment analysis data"""
    __tablename__ = 'sentiment_data'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False, index=True)
    date = Column(DateTime, nullable=False)
    sentiment_score = Column(Float)
    vader_compound = Column(Float)
    textblob_polarity = Column(Float)
    overall_sentiment = Column(String(20))
    tweet_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Composite index
    __table_args__ = (
        Index('idx_sentiment_symbol_date', 'symbol', 'date', unique=True),
    )
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'symbol': self.symbol,
            'date': self.date.isoformat() if self.date else None,
            'sentiment_score': self.sentiment_score,
            'vader_compound': self.vader_compound,
            'textblob_polarity': self.textblob_polarity,
            'overall_sentiment': self.overall_sentiment,
            'tweet_count': self.tweet_count
        }

class Recommendation(Base):
    """Model for investment recommendations"""
    __tablename__ = 'recommendations'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False, index=True)
    date = Column(DateTime, nullable=False, default=datetime.utcnow)
    recommendation = Column(String(10), nullable=False)  # 'BUY', 'SELL', 'HOLD'
    confidence = Column(Float)
    predicted_return_1d = Column(Float)
    predicted_return_7d = Column(Float)
    predicted_return_30d = Column(Float)
    model_version = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Index for latest recommendations
    __table_args__ = (
        Index('idx_recommendation_symbol_date', 'symbol', 'date'),
    )
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'symbol': self.symbol,
            'date': self.date.isoformat() if self.date else None,
            'recommendation': self.recommendation,
            'confidence': self.confidence,
            'predicted_returns': {
                '1_day': self.predicted_return_1d,
                '7_day': self.predicted_return_7d,
                '30_day': self.predicted_return_30d
            },
            'model_version': self.model_version
        }

class TechnicalIndicator(Base):
    """Model for technical indicators"""
    __tablename__ = 'technical_indicators'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), nullable=False, index=True)
    date = Column(DateTime, nullable=False)
    sma_7 = Column(Float)
    sma_30 = Column(Float)
    sma_90 = Column(Float)
    ema_12 = Column(Float)
    ema_26 = Column(Float)
    rsi_14 = Column(Float)
    macd = Column(Float)
    macd_signal = Column(Float)
    bb_upper = Column(Float)
    bb_middle = Column(Float)
    bb_lower = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Composite index
    __table_args__ = (
        Index('idx_indicator_symbol_date', 'symbol', 'date', unique=True),
    )