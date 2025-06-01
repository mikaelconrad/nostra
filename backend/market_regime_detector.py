"""
Simple Market Regime Detector for Cryptocurrency Trading Game
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict
from datetime import datetime

logger = logging.getLogger(__name__)

class MarketRegimeDetector:
    """Simple market regime detector"""
    
    def __init__(self, n_regimes: int = 3):
        self.n_regimes = n_regimes
        self.regime_labels = {0: 'Bear', 1: 'Sideways', 2: 'Bull'}
        self.volatility_labels = {0: 'Low_Vol', 1: 'Normal_Vol', 2: 'High_Vol'}
        
    def detect_trend_regimes(self, df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
        """Simple trend regime detection based on moving averages"""
        logger.info("Detecting trend-based market regimes")
        
        df_regimes = df.copy()
        df_regimes['price'] = df_regimes[price_col]
        
        # Simple moving averages
        df_regimes['ma_20'] = df_regimes['price'].rolling(20).mean()
        df_regimes['ma_50'] = df_regimes['price'].rolling(50).mean()
        
        # Simple trend classification
        conditions = [
            df_regimes['price'] < df_regimes['ma_20'] * 0.95,  # Bear
            df_regimes['price'] > df_regimes['ma_20'] * 1.05,  # Bull
        ]
        choices = [0, 2]
        df_regimes['trend_regime'] = np.select(conditions, choices, default=1)  # Sideways
        
        df_regimes['trend_regime_label'] = df_regimes['trend_regime'].map(self.regime_labels)
        
        return df_regimes
    
    def detect_volatility_regimes(self, df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
        """Simple volatility regime detection"""
        logger.info("Detecting volatility-based market regimes")
        
        df_regimes = df.copy()
        
        # Calculate volatility
        returns = df_regimes[price_col].pct_change()
        volatility = returns.rolling(20).std()
        
        # Simple volatility classification
        vol_75 = volatility.quantile(0.75)
        vol_25 = volatility.quantile(0.25)
        
        conditions = [
            volatility < vol_25,  # Low vol
            volatility > vol_75,  # High vol
        ]
        choices = [0, 2]
        df_regimes['volatility_regime'] = np.select(conditions, choices, default=1)  # Normal vol
        
        df_regimes['volatility_regime_label'] = df_regimes['volatility_regime'].map(self.volatility_labels)
        
        return df_regimes
    
    def detect_combined_regimes(self, df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
        """Detect combined market regimes"""
        logger.info("Detecting combined market regimes")
        
        df_combined = self.detect_trend_regimes(df, price_col)
        df_combined = self.detect_volatility_regimes(df_combined, price_col)
        
        # Create combined regime
        df_combined['combined_regime'] = (
            df_combined['trend_regime'].astype(str) + '_' + 
            df_combined['volatility_regime'].astype(str)
        )
        
        return df_combined

# Singleton instance
_regime_detector = None

def get_regime_detector(n_regimes: int = 3) -> MarketRegimeDetector:
    """Get singleton market regime detector instance"""
    global _regime_detector
    if _regime_detector is None:
        _regime_detector = MarketRegimeDetector(n_regimes=n_regimes)
    return _regime_detector