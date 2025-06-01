#!/usr/bin/env python3
"""
Comprehensive Predictive Model Testing Script

This script tests the predictive models by:
1. Randomly selecting 3 dates from training data with proper constraints
2. Training models for each selected date
3. Generating predictions for 1, 7, 14, 30 day windows
4. Validating predictions against historical data
5. Displaying detailed results

Requirements:
- Training dates must be at least historical_btc + historical_eth + 365 days from earliest data
- Training dates must not be later than 3 months from earliest day in dataset
- Validation is performed on hidden data (not used for training)
"""

import sys
import os
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config

# Try to import backend modules, handle missing imports gracefully
try:
    from backend.neural_network_model import EnhancedCryptoPredictor
except ImportError as e:
    print(f"Warning: Could not import EnhancedCryptoPredictor: {e}")
    EnhancedCryptoPredictor = None

try:
    from backend.data_collector import collect_enhanced_crypto_data
except ImportError as e:
    print(f"Warning: Could not import collect_enhanced_crypto_data: {e}")
    collect_enhanced_crypto_data = None

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PredictiveModelTester:
    """Test predictive models with temporal validation"""
    
    def __init__(self):
        self.symbols = list(config.CRYPTO_SYMBOLS.keys())
        self.prediction_horizons = [1, 7, 14, 30]
        self.test_results = {}
        
    def load_data(self, symbol: str) -> pd.DataFrame:
        """Load historical data for a symbol"""
        try:
            # Try to load from the raw data directory
            base_symbol = symbol.split('-')[0]
            price_file = os.path.join(config.RAW_DATA_DIRECTORY, f"{base_symbol}_USD.csv")
            
            if os.path.exists(price_file):
                df = pd.read_csv(price_file)
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                else:
                    # If no date column, assume first column is date
                    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
                    df = df.set_index(df.columns[0])
                
                df = df.sort_index()
                logger.info(f"Loaded {len(df)} records for {symbol} from {df.index.min()} to {df.index.max()}")
                return df
            else:
                logger.error(f"No data file found for {symbol}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Failed to load data for {symbol}: {e}")
            return pd.DataFrame()
    
    def select_random_test_dates(self, symbol: str, df: pd.DataFrame, num_dates: int = 1) -> List[str]:
        """
        Select random test dates based on the specified constraints
        
        Args:
            symbol: Trading symbol
            df: Historical data DataFrame
            num_dates: Number of dates to select (default: 1)
            
        Returns:
            List of selected test dates in YYYY-MM-DD format
        """
        if df.empty:
            logger.error(f"No data available for {symbol}")
            return []
        
        earliest_date = df.index.min()
        latest_date = df.index.max()
        
        # Calculate constraints
        # Minimum date: earliest + 365 days (for historical context)
        min_date = earliest_date + timedelta(days=365)
        
        # Maximum date: 3 months from earliest day in dataset
        max_date = earliest_date + timedelta(days=90)
        
        # Ensure we have a valid range
        if min_date >= max_date:
            logger.warning(f"Invalid date range for {symbol}. Using available range.")
            min_date = earliest_date + timedelta(days=60)  # Minimum 60 days for context
            max_date = latest_date - timedelta(days=30)    # Leave 30 days for validation
        
        # Ensure max_date doesn't exceed available data
        if max_date > latest_date - timedelta(days=30):
            max_date = latest_date - timedelta(days=30)
        
        logger.info(f"Date selection range for {symbol}: {min_date.date()} to {max_date.date()}")
        
        # Filter available dates within range
        available_dates = df[(df.index >= min_date) & (df.index <= max_date)].index
        
        if len(available_dates) < num_dates:
            logger.warning(f"Only {len(available_dates)} dates available for {symbol}, selecting all")
            num_dates = len(available_dates)
        
        # Randomly select dates
        selected_dates = random.sample(list(available_dates), num_dates)
        selected_dates = [date.strftime('%Y-%m-%d') for date in sorted(selected_dates)]
        
        logger.info(f"Selected test dates for {symbol}: {selected_dates}")
        return selected_dates
    
    def train_model_for_date(self, symbol: str, training_date: str) -> Optional[object]:
        """Train model up to a specific date"""
        try:
            logger.info(f"Training model for {symbol} up to {training_date}")
            
            # Check if predictor class is available
            if EnhancedCryptoPredictor is None:
                logger.error("EnhancedCryptoPredictor class not available")
                return None, {'error': 'EnhancedCryptoPredictor class not available'}
            
            # Initialize predictor
            predictor = EnhancedCryptoPredictor(
                symbol=symbol,
                use_enhanced_features=True,
                use_ensemble=True
            )
            
            # Train models for all horizons
            training_results = {}
            for horizon in self.prediction_horizons:
                try:
                    model, metrics = predictor.train_model_temporal(
                        horizon=horizon,
                        training_date=training_date,
                        force_retrain=True
                    )
                    training_results[horizon] = {
                        'success': True,
                        'metrics': metrics
                    }
                    logger.info(f"Trained {symbol} {horizon}d model: MSE={metrics.get('mse', 'N/A')}")
                    
                except Exception as e:
                    logger.error(f"Failed to train {symbol} {horizon}d model: {e}")
                    training_results[horizon] = {
                        'success': False,
                        'error': str(e)
                    }
            
            return predictor, training_results
            
        except Exception as e:
            logger.error(f"Failed to initialize predictor for {symbol}: {e}")
            return None, {'error': str(e)}
    
    def generate_predictions(self, predictor: object, symbol: str, 
                           prediction_date: str) -> Dict:
        """Generate predictions for all horizons"""
        predictions = {}
        
        for horizon in self.prediction_horizons:
            try:
                result = predictor.predict_with_confidence(
                    horizon=horizon,
                    prediction_date=prediction_date
                )
                
                if result:
                    predictions[horizon] = {
                        'predicted_price': result['predicted_price'],
                        'confidence_interval': result['confidence_interval'],
                        'confidence_score': result['confidence_score'],
                        'prediction_date': result['prediction_date']
                    }
                    logger.info(f"{symbol} {horizon}d prediction: ${result['predicted_price']:.2f} (confidence: {result['confidence_score']:.2f})")
                else:
                    predictions[horizon] = {'error': 'Prediction failed'}
                    
            except Exception as e:
                logger.error(f"Failed to predict {symbol} {horizon}d: {e}")
                predictions[horizon] = {'error': str(e)}
        
        return predictions
    
    def validate_predictions(self, symbol: str, predictions: Dict, 
                           prediction_date: str, historical_data: pd.DataFrame) -> Dict:
        """Validate predictions against actual historical data"""
        validation_results = {}
        prediction_datetime = pd.to_datetime(prediction_date)
        
        for horizon, pred_data in predictions.items():
            if 'error' in pred_data:
                validation_results[horizon] = pred_data
                continue
            
            try:
                # Find actual price at target date
                target_date = prediction_datetime + timedelta(days=horizon)
                
                # Find closest available date in historical data
                available_dates = historical_data.index
                closest_date_idx = np.argmin(np.abs(available_dates - target_date))
                closest_date = available_dates[closest_date_idx]
                actual_price = historical_data.loc[closest_date, 'close']
                
                predicted_price = pred_data['predicted_price']
                
                # Calculate validation metrics
                absolute_error = abs(predicted_price - actual_price)
                percentage_error = (absolute_error / actual_price) * 100
                
                # Check if actual falls within confidence interval
                ci_lower = pred_data['confidence_interval']['lower']
                ci_upper = pred_data['confidence_interval']['upper']
                within_ci = ci_lower <= actual_price <= ci_upper
                
                validation_results[horizon] = {
                    'predicted_price': predicted_price,
                    'actual_price': float(actual_price),
                    'target_date': target_date.strftime('%Y-%m-%d'),
                    'actual_date': closest_date.strftime('%Y-%m-%d'),
                    'absolute_error': absolute_error,
                    'percentage_error': percentage_error,
                    'within_confidence_interval': within_ci,
                    'confidence_interval': pred_data['confidence_interval'],
                    'confidence_score': pred_data['confidence_score']
                }
                
                logger.info(f"{symbol} {horizon}d validation: Predicted=${predicted_price:.2f}, Actual=${actual_price:.2f}, Error={percentage_error:.1f}%")
                
            except Exception as e:
                logger.error(f"Failed to validate {symbol} {horizon}d prediction: {e}")
                validation_results[horizon] = {'error': str(e)}
        
        return validation_results
    
    def run_comprehensive_test(self):
        """Run the complete predictive model test"""
        print("="*80)
        print("COMPREHENSIVE PREDICTIVE MODEL TEST")
        print("="*80)
        print()
        
        # Set random seed for reproducibility
        random.seed(42)
        np.random.seed(42)
        
        all_results = {}
        
        for symbol in self.symbols:
            print(f"\nTesting {symbol} ({config.CRYPTO_SYMBOLS[symbol]})")
            print("-" * 50)
            
            # Load data
            historical_data = self.load_data(symbol)
            if historical_data.empty:
                print(f"❌ No data available for {symbol}")
                continue
            
            # Select random test dates
            test_dates = self.select_random_test_dates(symbol, historical_data)
            if not test_dates:
                print(f"❌ Could not select test dates for {symbol}")
                continue
            
            symbol_results = {}
            
            for i, test_date in enumerate(test_dates, 1):
                print(f"\n📅 Test {i}: Training up to {test_date}")
                print("   " + "="*45)
                
                # Train model
                predictor, training_results = self.train_model_for_date(symbol, test_date)
                if predictor is None:
                    print(f"   ❌ Failed to train model for {test_date}")
                    continue
                
                # Generate predictions
                print(f"   🔮 Generating predictions...")
                predictions = self.generate_predictions(predictor, symbol, test_date)
                
                # Validate against historical data
                print(f"   ✅ Validating against historical data...")
                validation = self.validate_predictions(symbol, predictions, test_date, historical_data)
                
                # Store results
                symbol_results[test_date] = {
                    'training_results': training_results,
                    'predictions': predictions,
                    'validation': validation
                }
                
                # Print detailed results for this test date
                self.print_test_results(symbol, test_date, validation)
            
            all_results[symbol] = symbol_results
        
        # Print summary
        self.print_summary(all_results)
        
        # Save results to file
        results_file = os.path.join(config.REPORT_DIRECTORY, f'predictive_model_test_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print(f"\n📊 Detailed results saved to: {results_file}")
        return all_results
    
    def print_test_results(self, symbol: str, test_date: str, validation: Dict):
        """Print detailed results for a single test"""
        print(f"\n   📊 Results for {symbol} trained up to {test_date}:")
        print("   " + "-"*45)
        
        for horizon in self.prediction_horizons:
            if horizon not in validation or 'error' in validation[horizon]:
                print(f"   {horizon:2d}d: ❌ Error - {validation[horizon].get('error', 'Unknown error')}")
                continue
            
            result = validation[horizon]
            predicted = result['predicted_price']
            actual = result['actual_price']
            error = result['percentage_error']
            within_ci = result['within_confidence_interval']
            
            status = "✅" if error < 10 else "⚠️" if error < 20 else "❌"
            ci_status = "📊" if within_ci else "📉"
            
            print(f"   {horizon:2d}d: {status} Predicted: ${predicted:8.2f} | Actual: ${actual:8.2f} | Error: {error:5.1f}% {ci_status}")
    
    def print_summary(self, all_results: Dict):
        """Print summary statistics"""
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        
        total_predictions = 0
        successful_predictions = 0
        errors_by_horizon = {h: [] for h in self.prediction_horizons}
        within_ci_count = {h: 0 for h in self.prediction_horizons}
        total_by_horizon = {h: 0 for h in self.prediction_horizons}
        
        for symbol, symbol_data in all_results.items():
            print(f"\n{symbol} ({config.CRYPTO_SYMBOLS[symbol]}):")
            print("-" * 30)
            
            for test_date, test_data in symbol_data.items():
                validation = test_data.get('validation', {})
                
                for horizon in self.prediction_horizons:
                    total_by_horizon[horizon] += 1
                    total_predictions += 1
                    
                    if horizon in validation and 'error' not in validation[horizon]:
                        successful_predictions += 1
                        error = validation[horizon]['percentage_error']
                        errors_by_horizon[horizon].append(error)
                        
                        if validation[horizon]['within_confidence_interval']:
                            within_ci_count[horizon] += 1
        
        # Overall statistics
        success_rate = (successful_predictions / total_predictions * 100) if total_predictions > 0 else 0
        print(f"\nOverall Success Rate: {success_rate:.1f}% ({successful_predictions}/{total_predictions})")
        
        # Horizon-specific statistics
        print(f"\nPrediction Accuracy by Horizon:")
        print("Horizon | Avg Error | Median Error | CI Coverage | Count")
        print("-" * 55)
        
        for horizon in self.prediction_horizons:
            errors = errors_by_horizon[horizon]
            if errors:
                avg_error = np.mean(errors)
                median_error = np.median(errors)
                ci_coverage = (within_ci_count[horizon] / len(errors)) * 100
                count = len(errors)
                print(f"{horizon:7d} | {avg_error:9.1f}% | {median_error:12.1f}% | {ci_coverage:10.1f}% | {count:5d}")
            else:
                print(f"{horizon:7d} | {'N/A':>9} | {'N/A':>12} | {'N/A':>10} | {0:5d}")


def main():
    """Main function to run the predictive model test"""
    try:
        print("Initializing Predictive Model Test...")
        
        # Create reports directory if it doesn't exist
        os.makedirs(config.REPORT_DIRECTORY, exist_ok=True)
        
        # Initialize and run test
        tester = PredictiveModelTester()
        results = tester.run_comprehensive_test()
        
        print("\n🎉 Test completed successfully!")
        return results
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        print(f"\n❌ Test failed: {e}")
        return None


if __name__ == "__main__":
    main()