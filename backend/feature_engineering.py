"""
Advanced Feature Engineering Pipeline for Cryptocurrency Prediction
Implements market microstructure, cross-asset correlation, technical indicators,
time-based cyclical features, and market regime indicators
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ta
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

import config

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Advanced feature engineering for cryptocurrency data"""
    
    def __init__(self, max_features: int = 30):
        self.scalers = {}
        self.feature_names = []
        self.feature_importance = {}
        self.max_features = max_features
        self.feature_selector = None
        self.selected_features = []
        
    def create_market_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create market microstructure features
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with microstructure features
        """
        logger.info("Creating market microstructure features")
        
        df_features = df.copy()
        
        # Price-based microstructure features
        df_features['price_range'] = df_features['high'] - df_features['low']
        df_features['price_range_pct'] = df_features['price_range'] / df_features['close']
        
        # Body and shadow ratios (candlestick analysis)
        df_features['body_size'] = abs(df_features['close'] - df_features['open'])
        df_features['upper_shadow'] = df_features['high'] - np.maximum(df_features['close'], df_features['open'])
        df_features['lower_shadow'] = np.minimum(df_features['close'], df_features['open']) - df_features['low']
        
        df_features['body_ratio'] = df_features['body_size'] / df_features['price_range']
        df_features['upper_shadow_ratio'] = df_features['upper_shadow'] / df_features['price_range']
        df_features['lower_shadow_ratio'] = df_features['lower_shadow'] / df_features['price_range']
        
        # Volume-price relationships
        df_features['volume_price_trend'] = df_features['volume'] * np.sign(df_features['close'].pct_change())
        df_features['price_volume_correlation'] = df_features['close'].rolling(20).corr(df_features['volume'])
        
        # VWAP and VWAP deviations
        df_features['vwap'] = (df_features['volume'] * (df_features['high'] + df_features['low'] + df_features['close']) / 3).cumsum() / df_features['volume'].cumsum()
        df_features['vwap_deviation'] = (df_features['close'] - df_features['vwap']) / df_features['vwap']
        
        # Tick direction and momentum
        df_features['tick_direction'] = np.sign(df_features['close'].diff())
        df_features['tick_momentum'] = df_features['tick_direction'].rolling(5).sum()
        
        # Intraday patterns (if available)
        if 'timestamp' in df_features.columns:
            df_features['timestamp'] = pd.to_datetime(df_features['timestamp'])
            df_features['hour'] = df_features['timestamp'].dt.hour
            df_features['day_of_week'] = df_features['timestamp'].dt.dayofweek
        
        # Liquidity proxies
        df_features['bid_ask_spread_proxy'] = df_features['price_range'] / df_features['volume']
        df_features['market_impact'] = df_features['close'].pct_change() / np.log1p(df_features['volume'])
        
        # Price clustering and round number effects
        df_features['price_round_10'] = (df_features['close'] % 10 == 0).astype(int)
        df_features['price_round_100'] = (df_features['close'] % 100 == 0).astype(int)
        df_features['price_round_1000'] = (df_features['close'] % 1000 == 0).astype(int)
        
        logger.info(f"Created {len([col for col in df_features.columns if col not in df.columns])} microstructure features")
        return df_features
    
    def create_cross_asset_correlation_features(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Create cross-asset correlation features
        
        Args:
            data_dict: Dictionary with symbol -> DataFrame mapping
            
        Returns:
            Dictionary with enhanced DataFrames including correlation features
        """
        logger.info("Creating cross-asset correlation features")
        
        symbols = list(data_dict.keys())
        enhanced_data = {}
        
        # Create correlation matrices for different time windows
        windows = [5, 10, 20, 50]
        
        for symbol in symbols:
            df = data_dict[symbol].copy()
            
            # Get returns for correlation calculation
            returns_data = {}
            for s in symbols:
                if s in data_dict and not data_dict[s].empty:
                    returns_data[s] = data_dict[s]['close'].pct_change()
            
            if len(returns_data) > 1:
                for window in windows:
                    for other_symbol in symbols:
                        if other_symbol != symbol and other_symbol in returns_data:
                            # Rolling correlation
                            correlation = returns_data[symbol].rolling(window).corr(returns_data[other_symbol])
                            df[f'corr_{other_symbol.replace("-", "_")}_{window}d'] = correlation
                            
                            # Correlation volatility
                            df[f'corr_vol_{other_symbol.replace("-", "_")}_{window}d'] = correlation.rolling(window).std()
            
            # Market beta (correlation with overall market)
            if len(returns_data) > 1:
                market_returns = pd.concat(returns_data.values(), axis=1).mean(axis=1)
                for window in windows:
                    df[f'market_beta_{window}d'] = returns_data[symbol].rolling(window).corr(market_returns)
            
            # Lead-lag relationships
            for other_symbol in symbols:
                if other_symbol != symbol and other_symbol in returns_data:
                    # Lead correlation (current returns vs future other returns)
                    df[f'lead_corr_{other_symbol.replace("-", "_")}'] = returns_data[symbol].rolling(20).corr(returns_data[other_symbol].shift(-1))
                    
                    # Lag correlation (current returns vs past other returns)
                    df[f'lag_corr_{other_symbol.replace("-", "_")}'] = returns_data[symbol].rolling(20).corr(returns_data[other_symbol].shift(1))
            
            enhanced_data[symbol] = df
        
        logger.info(f"Created correlation features for {len(symbols)} assets")
        return enhanced_data
    
    def create_advanced_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced technical indicators beyond basic TA
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with advanced technical indicators
        """
        logger.info("Creating advanced technical indicators")
        
        df_features = df.copy()
        
        # Ensure we have required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df_features.columns:
                logger.warning(f"Missing required column: {col}")
                return df_features
        
        # Basic momentum indicators
        df_features['rsi_14'] = ta.momentum.RSIIndicator(df_features['close'], window=14).rsi()
        df_features['rsi_30'] = ta.momentum.RSIIndicator(df_features['close'], window=30).rsi()
        
        # MACD variations
        macd = ta.trend.MACD(df_features['close'])
        df_features['macd'] = macd.macd()
        df_features['macd_signal'] = macd.macd_signal()
        df_features['macd_histogram'] = macd.macd_diff()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df_features['close'])
        df_features['bb_upper'] = bb.bollinger_hband()
        df_features['bb_lower'] = bb.bollinger_lband()
        df_features['bb_middle'] = bb.bollinger_mavg()
        df_features['bb_width'] = (df_features['bb_upper'] - df_features['bb_lower']) / df_features['bb_middle']
        df_features['bb_position'] = (df_features['close'] - df_features['bb_lower']) / (df_features['bb_upper'] - df_features['bb_lower'])
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(df_features['high'], df_features['low'], df_features['close'])
        df_features['stoch_k'] = stoch.stoch()
        df_features['stoch_d'] = stoch.stoch_signal()
        
        # Williams %R
        df_features['williams_r'] = ta.momentum.WilliamsRIndicator(df_features['high'], df_features['low'], df_features['close']).williams_r()
        
        # Average True Range and volatility
        df_features['atr'] = ta.volatility.AverageTrueRange(df_features['high'], df_features['low'], df_features['close']).average_true_range()
        df_features['atr_pct'] = df_features['atr'] / df_features['close']
        
        # Commodity Channel Index
        df_features['cci'] = ta.trend.CCIIndicator(df_features['high'], df_features['low'], df_features['close']).cci()
        
        # Money Flow Index
        df_features['mfi'] = ta.volume.MFIIndicator(df_features['high'], df_features['low'], df_features['close'], df_features['volume']).money_flow_index()
        
        # On Balance Volume
        df_features['obv'] = ta.volume.OnBalanceVolumeIndicator(df_features['close'], df_features['volume']).on_balance_volume()
        df_features['obv_pct_change'] = df_features['obv'].pct_change()
        
        # Parabolic SAR
        df_features['sar'] = ta.trend.PSARIndicator(df_features['high'], df_features['low'], df_features['close']).psar()
        df_features['sar_trend'] = (df_features['close'] > df_features['sar']).astype(int)
        
        # Ichimoku Cloud components
        ichimoku = ta.trend.IchimokuIndicator(df_features['high'], df_features['low'])
        df_features['ichimoku_a'] = ichimoku.ichimoku_a()
        df_features['ichimoku_b'] = ichimoku.ichimoku_b()
        df_features['ichimoku_base'] = ichimoku.ichimoku_base_line()
        df_features['ichimoku_conversion'] = ichimoku.ichimoku_conversion_line()
        
        # Custom indicators
        # Price momentum oscillator
        df_features['price_momentum_5'] = df_features['close'].pct_change(5)
        df_features['price_momentum_10'] = df_features['close'].pct_change(10)
        df_features['price_momentum_20'] = df_features['close'].pct_change(20)
        
        # Volume momentum
        df_features['volume_momentum_5'] = df_features['volume'].pct_change(5)
        df_features['volume_momentum_10'] = df_features['volume'].pct_change(10)
        
        # Price acceleration
        returns = df_features['close'].pct_change()
        df_features['price_acceleration'] = returns.diff()
        
        # Volatility indicators
        df_features['volatility_5'] = returns.rolling(5).std()
        df_features['volatility_10'] = returns.rolling(10).std()
        df_features['volatility_20'] = returns.rolling(20).std()
        
        # Skewness and kurtosis of returns
        df_features['returns_skew_20'] = returns.rolling(20).skew()
        df_features['returns_kurtosis_20'] = returns.rolling(20).kurt()
        
        # Support and resistance levels
        df_features['resistance_20'] = df_features['high'].rolling(20).max()
        df_features['support_20'] = df_features['low'].rolling(20).min()
        df_features['resistance_distance'] = (df_features['resistance_20'] - df_features['close']) / df_features['close']
        df_features['support_distance'] = (df_features['close'] - df_features['support_20']) / df_features['close']
        
        logger.info(f"Created {len([col for col in df_features.columns if col not in df.columns])} technical indicators")
        return df_features
    
    def create_time_based_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based cyclical features
        
        Args:
            df: DataFrame with timestamp information
            
        Returns:
            DataFrame with time-based features
        """
        logger.info("Creating time-based cyclical features")
        
        df_features = df.copy()
        
        # Ensure we have a datetime index or column
        if 'timestamp' in df_features.columns:
            df_features['timestamp'] = pd.to_datetime(df_features['timestamp'])
            dt_col = df_features['timestamp']
            use_dt_accessor = True
        elif isinstance(df_features.index, pd.DatetimeIndex):
            dt_col = df_features.index
            use_dt_accessor = False
        else:
            logger.warning("No datetime information found for time-based features")
            return df_features
        
        # Day of week cyclical encoding
        day_of_week = dt_col.dayofweek
        df_features['day_of_week_sin'] = np.sin(2 * np.pi * day_of_week / 7)
        df_features['day_of_week_cos'] = np.cos(2 * np.pi * day_of_week / 7)
        
        # Month cyclical encoding
        month = dt_col.month
        df_features['month_sin'] = np.sin(2 * np.pi * month / 12)
        df_features['month_cos'] = np.cos(2 * np.pi * month / 12)
        
        # Quarter cyclical encoding
        quarter = dt_col.quarter
        df_features['quarter_sin'] = np.sin(2 * np.pi * quarter / 4)
        df_features['quarter_cos'] = np.cos(2 * np.pi * quarter / 4)
        
        # Year progress (0-1)
        if use_dt_accessor:
            year_start = pd.to_datetime(dt_col.dt.year.astype(str) + '-01-01')
            year_end = pd.to_datetime((dt_col.dt.year + 1).astype(str) + '-01-01')
        else:
            year_start = pd.to_datetime(dt_col.year.astype(str) + '-01-01')
            year_end = pd.to_datetime((dt_col.year + 1).astype(str) + '-01-01')
        df_features['year_progress'] = (dt_col - year_start) / (year_end - year_start)
        
        # Weekend indicator
        df_features['is_weekend'] = (day_of_week >= 5).astype(int)
        
        # Month end/beginning indicators
        if use_dt_accessor:
            df_features['is_month_start'] = (dt_col.dt.day <= 3).astype(int)
            df_features['is_month_end'] = (dt_col.dt.day >= dt_col.dt.daysinmonth - 2).astype(int)
            
            # Quarter end/beginning indicators
            df_features['is_quarter_start'] = ((dt_col.dt.month % 3 == 1) & (dt_col.dt.day <= 5)).astype(int)
            df_features['is_quarter_end'] = ((dt_col.dt.month % 3 == 0) & (dt_col.dt.day >= 25)).astype(int)
            
            # Holiday season indicators (simplified)
            df_features['is_holiday_season'] = (
                ((dt_col.dt.month == 12) & (dt_col.dt.day >= 20)) |
                ((dt_col.dt.month == 1) & (dt_col.dt.day <= 10))
            ).astype(int)
        else:
            df_features['is_month_start'] = (dt_col.day <= 3).astype(int)
            df_features['is_month_end'] = (dt_col.day >= dt_col.daysinmonth - 2).astype(int)
            
            # Quarter end/beginning indicators
            df_features['is_quarter_start'] = ((dt_col.month % 3 == 1) & (dt_col.day <= 5)).astype(int)
            df_features['is_quarter_end'] = ((dt_col.month % 3 == 0) & (dt_col.day >= 25)).astype(int)
            
            # Holiday season indicators (simplified)
            df_features['is_holiday_season'] = (
                ((dt_col.month == 12) & (dt_col.day >= 20)) |
                ((dt_col.month == 1) & (dt_col.day <= 10))
            ).astype(int)
        
        # Business day indicators
        df_features['is_business_day'] = pd.bdate_range(start=dt_col.min(), end=dt_col.max()).intersection(dt_col).size > 0
        
        # Days since events (if we have price data)
        if 'close' in df_features.columns:
            # Days since local maximum
            peaks, _ = find_peaks(df_features['close'], distance=5)
            df_features['days_since_peak'] = np.nan
            for i, peak in enumerate(peaks):
                df_features.iloc[peak:, df_features.columns.get_loc('days_since_peak')] = range(len(df_features) - peak)
            df_features['days_since_peak'].fillna(method='ffill', inplace=True)
            
            # Days since local minimum
            troughs, _ = find_peaks(-df_features['close'], distance=5)
            df_features['days_since_trough'] = np.nan
            for i, trough in enumerate(troughs):
                df_features.iloc[trough:, df_features.columns.get_loc('days_since_trough')] = range(len(df_features) - trough)
            df_features['days_since_trough'].fillna(method='ffill', inplace=True)
        
        logger.info(f"Created {len([col for col in df_features.columns if col not in df.columns])} time-based features")
        return df_features
    
    def create_regime_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create market regime indicators
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with regime indicators
        """
        logger.info("Creating market regime indicators")
        
        df_features = df.copy()
        
        if 'close' not in df_features.columns:
            logger.warning("No close price data for regime indicators")
            return df_features
        
        returns = df_features['close'].pct_change()
        
        # Volatility regime
        volatility = returns.rolling(20).std()
        vol_mean = volatility.rolling(100).mean()
        vol_std = volatility.rolling(100).std()
        
        df_features['volatility_regime'] = np.where(
            volatility > vol_mean + vol_std, 2,  # High volatility
            np.where(volatility < vol_mean - vol_std, 0, 1)  # Low, Normal volatility
        )
        
        # Trend regime
        ma_short = df_features['close'].rolling(20).mean()
        ma_long = df_features['close'].rolling(50).mean()
        
        df_features['trend_regime'] = np.where(
            ma_short > ma_long, 1,  # Uptrend
            np.where(ma_short < ma_long, -1, 0)  # Downtrend, Sideways
        )
        
        # Mean reversion regime
        z_score = (df_features['close'] - df_features['close'].rolling(20).mean()) / df_features['close'].rolling(20).std()
        df_features['mean_reversion_regime'] = (abs(z_score) > 2).astype(int)
        
        # Momentum regime
        momentum_5 = df_features['close'].pct_change(5)
        momentum_20 = df_features['close'].pct_change(20)
        
        df_features['momentum_regime'] = np.where(
            (momentum_5 > 0) & (momentum_20 > 0), 2,  # Strong momentum
            np.where((momentum_5 < 0) & (momentum_20 < 0), 0, 1)  # Weak momentum, Mixed
        )
        
        # Market stress regime (based on volatility and correlation)
        df_features['market_stress'] = (
            (df_features['volatility_regime'] == 2) & 
            (abs(returns) > returns.rolling(50).quantile(0.95))
        ).astype(int)
        
        # Bull/Bear market regime (longer term)
        ma_200 = df_features['close'].rolling(200).mean()
        df_features['bull_bear_regime'] = np.where(
            df_features['close'] > ma_200, 1, 0  # Bull, Bear
        )
        
        # Regime stability (how long in current regime)
        for regime_col in ['volatility_regime', 'trend_regime', 'momentum_regime']:
            regime_changes = df_features[regime_col].diff() != 0
            df_features[f'{regime_col}_stability'] = (~regime_changes).cumsum()
        
        logger.info(f"Created {len([col for col in df_features.columns if col not in df.columns])} regime indicators")
        return df_features
    
    def apply_feature_engineering(self, data_dict: Dict[str, pd.DataFrame], 
                                include_cross_asset: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Apply complete feature engineering pipeline
        
        Args:
            data_dict: Dictionary with symbol -> DataFrame mapping
            include_cross_asset: Whether to include cross-asset correlation features
            
        Returns:
            Dictionary with enhanced DataFrames
        """
        logger.info("Starting complete feature engineering pipeline")
        
        enhanced_data = {}
        
        # Step 1: Apply individual asset features
        for symbol, df in data_dict.items():
            if df.empty:
                logger.warning(f"Empty DataFrame for {symbol}, skipping")
                enhanced_data[symbol] = df
                continue
            
            logger.info(f"Processing features for {symbol}")
            
            # Market microstructure features
            enhanced_df = self.create_market_microstructure_features(df)
            
            # Advanced technical indicators
            enhanced_df = self.create_advanced_technical_indicators(enhanced_df)
            
            # Time-based features
            enhanced_df = self.create_time_based_features(enhanced_df)
            
            # Regime indicators
            enhanced_df = self.create_regime_indicators(enhanced_df)
            
            enhanced_data[symbol] = enhanced_df
        
        # Step 2: Cross-asset correlation features
        if include_cross_asset and len(enhanced_data) > 1:
            enhanced_data = self.create_cross_asset_correlation_features(enhanced_data)
        
        # Step 3: Feature selection and importance tracking
        self._update_feature_names(enhanced_data)
        
        # Step 3: Feature selection to reduce overfitting
        if self.max_features and len(self.feature_names) > self.max_features:
            enhanced_data = self._apply_feature_selection(enhanced_data)
        
        logger.info(f"Feature engineering completed for {len(enhanced_data)} assets")
        logger.info(f"Selected features per asset: {len(self.selected_features or self.feature_names)}")
        
        return enhanced_data
    
    def _update_feature_names(self, data_dict: Dict[str, pd.DataFrame]):
        """Update the list of feature names"""
        if data_dict:
            sample_df = list(data_dict.values())[0]
            self.feature_names = [col for col in sample_df.columns 
                                if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'date']]
        
        logger.info(f"Tracked {len(self.feature_names)} engineered features")
    
    def _apply_feature_selection(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Apply feature selection to reduce dimensionality and prevent overfitting
        
        Args:
            data_dict: Dictionary with symbol -> DataFrame mapping
            
        Returns:
            Dictionary with reduced feature DataFrames
        """
        logger.info(f"Applying feature selection: {len(self.feature_names)} -> {self.max_features} features")
        
        # Use the first available dataset for feature selection
        sample_symbol = list(data_dict.keys())[0]
        sample_df = data_dict[sample_symbol]
        
        if 'close' not in sample_df.columns:
            logger.warning("No target column 'close' for feature selection")
            return data_dict
        
        # Prepare data for feature selection
        X = sample_df[self.feature_names].fillna(0)
        y = sample_df['close'].pct_change().shift(-1).fillna(0)  # Next period return
        
        # Remove rows with infinite or very large values
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X = X[mask]
        y = y[mask]
        
        if len(X) < 100:  # Need sufficient data for feature selection
            logger.warning("Insufficient data for feature selection")
            return data_dict
        
        try:
            # Combine multiple feature selection methods
            # 1. Mutual information (captures non-linear relationships)
            mi_selector = SelectKBest(score_func=mutual_info_regression, k=min(self.max_features * 2, len(self.feature_names)))
            X_mi = mi_selector.fit_transform(X, y)
            mi_features = [self.feature_names[i] for i in mi_selector.get_support(indices=True)]
            
            # 2. F-regression (captures linear relationships)
            f_selector = SelectKBest(score_func=f_regression, k=min(self.max_features, len(mi_features)))
            X_f = f_selector.fit_transform(X[mi_features], y)
            selected_indices = f_selector.get_support(indices=True)
            self.selected_features = [mi_features[i] for i in selected_indices]
            
            # 3. Ensure we have essential OHLCV features
            essential_features = ['volume', 'rsi_14', 'macd', 'volatility_5', 'price_momentum_5']
            for feature in essential_features:
                if feature in self.feature_names and feature not in self.selected_features:
                    if len(self.selected_features) < self.max_features:
                        self.selected_features.append(feature)
                    else:
                        # Replace least important feature
                        self.selected_features[-1] = feature
            
            logger.info(f"Selected {len(self.selected_features)} features: {self.selected_features[:10]}...")
            
            # Apply feature selection to all datasets
            reduced_data = {}
            for symbol, df in data_dict.items():
                # Keep essential columns
                essential_cols = ['open', 'high', 'low', 'close', 'volume']
                available_essential = [col for col in essential_cols if col in df.columns]
                
                # Keep selected features that exist in this dataset
                available_selected = [col for col in self.selected_features if col in df.columns]
                
                # Combine essential and selected features
                keep_cols = list(set(available_essential + available_selected))
                reduced_data[symbol] = df[keep_cols].copy()
            
            return reduced_data
            
        except Exception as e:
            logger.error(f"Feature selection failed: {e}")
            return data_dict
    
    def get_feature_importance_summary(self) -> Dict:
        """
        Get summary of feature importance and categories
        
        Returns:
            Dictionary with feature categorization and importance
        """
        active_features = self.selected_features or self.feature_names
        
        categories = {
            'microstructure': [f for f in active_features if any(term in f.lower() for term in 
                             ['body', 'shadow', 'vwap', 'tick', 'spread', 'impact', 'round'])],
            'technical': [f for f in active_features if any(term in f.lower() for term in 
                        ['rsi', 'macd', 'bb_', 'stoch', 'williams', 'atr', 'cci', 'mfi', 'obv', 'sar', 'ichimoku'])],
            'correlation': [f for f in active_features if 'corr' in f.lower()],
            'temporal': [f for f in active_features if any(term in f.lower() for term in 
                       ['day_', 'month_', 'quarter_', 'year_', 'weekend', 'holiday', 'business'])],
            'regime': [f for f in active_features if 'regime' in f.lower()],
            'momentum': [f for f in active_features if 'momentum' in f.lower()],
            'volatility': [f for f in active_features if 'volatility' in f.lower() or 'vol_' in f.lower()]
        }
        
        return {
            'total_features_engineered': len(self.feature_names),
            'total_features_selected': len(active_features),
            'feature_reduction_ratio': len(active_features) / len(self.feature_names) if self.feature_names else 1,
            'categories': {cat: len(features) for cat, features in categories.items()},
            'selected_features': active_features,
            'feature_list': self.feature_names
        }


# Singleton instance
_feature_engineer = None

def get_feature_engineer(max_features: int = 30) -> FeatureEngineer:
    """Get singleton feature engineer instance"""
    global _feature_engineer
    if _feature_engineer is None:
        _feature_engineer = FeatureEngineer(max_features=max_features)
    return _feature_engineer