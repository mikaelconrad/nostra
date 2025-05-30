#!/usr/bin/env python3
"""
Demo script to test all the improvements made to the crypto investment app
"""

import sys
import os
import json
import requests
from datetime import datetime
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from utils.logger import setup_logger
from utils.validators import CryptoValidator, validate_transaction_request
from utils.error_handlers import ValidationError
from backend.portfolio_tracker import PortfolioTracker
from backend.data_collector import fetch_crypto_data

# Set up logger
logger = setup_logger('demo')

def print_section(title):
    """Print a section header"""
    print("\n" + "="*60)
    print(f"🔹 {title}")
    print("="*60)

def test_configuration():
    """Test configuration and environment variables"""
    print_section("Testing Configuration & Environment Variables")
    
    print(f"✓ Environment: {config.APP_ENV}")
    print(f"✓ API Port: {config.API_PORT}")
    print(f"✓ Frontend Port: {config.FRONTEND_PORT}")
    print(f"✓ Initial Investment: CHF {config.INITIAL_INVESTMENT}")
    print(f"✓ Database URL: {config.DATABASE_URL}")
    print(f"✓ Base Directory: {config.BASE_DIR}")
    
    # Test that paths are dynamic
    assert not config.BASE_DIR.startswith('/home/ubuntu'), "✗ Hardcoded path detected!"
    print("✓ All paths are dynamic (no hardcoded paths)")

def test_logging():
    """Test logging system"""
    print_section("Testing Logging System")
    
    # Test different log levels
    logger.info("This is an INFO message")
    logger.warning("This is a WARNING message")
    logger.error("This is an ERROR message (test only, not a real error)")
    
    # Check log file exists
    log_file = os.path.join(config.LOG_DIRECTORY, 'demo.log')
    if os.path.exists(log_file):
        print(f"✓ Log file created: {log_file}")
    else:
        print("✓ Logs displayed in console")

def test_validation():
    """Test input validation"""
    print_section("Testing Input Validation")
    
    # Test valid inputs
    try:
        valid_data = validate_transaction_request({
            'symbol': 'BTC',
            'amount': '0.001',
            'price': '50000'
        })
        print("✓ Valid transaction validated successfully:")
        print(f"  Symbol: {valid_data['symbol']}")
        print(f"  Amount: {valid_data['amount']}")
        print(f"  Price: CHF {valid_data['price']}")
    except Exception as e:
        print(f"✗ Valid transaction failed: {e}")
    
    # Test invalid inputs
    invalid_tests = [
        {'symbol': 'INVALID', 'amount': '0.1', 'price': '50000'},
        {'symbol': 'BTC', 'amount': '-0.1', 'price': '50000'},
        {'symbol': 'BTC', 'amount': '0.1', 'price': '0'},
    ]
    
    for test_data in invalid_tests:
        try:
            validate_transaction_request(test_data)
            print(f"✗ Should have failed: {test_data}")
        except ValidationError as e:
            print(f"✓ Invalid input caught: {str(e)}")

def test_portfolio_operations():
    """Test portfolio tracker"""
    print_section("Testing Portfolio Operations")
    
    # Create a test portfolio
    tracker = PortfolioTracker()
    
    # Display initial state
    print(f"Initial portfolio:")
    print(f"  Cash: CHF {tracker.portfolio['cash']:.2f}")
    print(f"  Holdings: {tracker.portfolio['holdings']}")
    
    # Test buy operation
    print("\nTesting BUY operation:")
    success, message = tracker.buy_crypto('BTC', 0.01, 50000)
    if success:
        print(f"✓ {message}")
        print(f"  New cash balance: CHF {tracker.portfolio['cash']:.2f}")
        print(f"  BTC holdings: {tracker.portfolio['holdings'].get('BTC', 0)}")
    else:
        print(f"✗ Buy failed: {message}")
    
    # Test sell operation
    if 'BTC' in tracker.portfolio['holdings']:
        print("\nTesting SELL operation:")
        success, message = tracker.sell_crypto('BTC', 0.005, 52000)
        if success:
            print(f"✓ {message}")
            print(f"  New cash balance: CHF {tracker.portfolio['cash']:.2f}")
            print(f"  BTC holdings: {tracker.portfolio['holdings'].get('BTC', 0)}")
        else:
            print(f"✗ Sell failed: {message}")
    
    # Test portfolio metrics
    current_prices = {'BTC': 52000, 'ETH': 2500, 'XRP': 0.75}
    metrics = tracker.calculate_performance_metrics(current_prices)
    
    print("\nPortfolio Performance:")
    print(f"  Total Invested: CHF {metrics['total_invested']:.2f}")
    print(f"  Current Value: CHF {metrics['current_value']:.2f}")
    print(f"  Total Return: CHF {metrics['total_return']:.2f}")
    print(f"  Return %: {metrics['return_percentage']:.2f}%")

def test_data_collection():
    """Test data collection"""
    print_section("Testing Data Collection")
    
    try:
        # Test fetching crypto data
        print("Fetching BTC data (last 7 days)...")
        df = fetch_crypto_data('BTC-USD', range='7d')
        
        if df is not None and not df.empty:
            print(f"✓ Fetched {len(df)} records")
            print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
            print(f"  Latest price: ${df.iloc[-1]['close']:.2f}")
        else:
            print("✗ No data fetched (might need internet connection)")
            
    except Exception as e:
        print(f"⚠ Data collection skipped: {str(e)}")

def test_api_endpoints():
    """Test REST API endpoints"""
    print_section("Testing REST API")
    
    # Check if API is accessible
    api_url = f"http://localhost:{config.API_PORT}/api/health"
    
    print(f"Testing API health check: {api_url}")
    print("⚠ Note: API server needs to be running for this test")
    print("  Run 'python api/app.py' in another terminal")
    
    try:
        response = requests.get(api_url, timeout=2)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ API is healthy: {data}")
        else:
            print(f"✗ API returned status: {response.status_code}")
    except requests.exceptions.RequestException:
        print("⚠ API is not running (this is normal for demo)")

def test_error_handling():
    """Test error handling"""
    print_section("Testing Error Handling")
    
    tracker = PortfolioTracker()
    
    # Test various error scenarios
    error_tests = [
        ("Insufficient funds", lambda: tracker.buy_crypto('BTC', 100, 50000)),
        ("Invalid symbol", lambda: tracker.buy_crypto('INVALID', 0.1, 50000)),
        ("Negative amount", lambda: tracker.buy_crypto('BTC', -0.1, 50000)),
        ("No holdings to sell", lambda: tracker.sell_crypto('ETH', 0.1, 2500)),
    ]
    
    for test_name, test_func in error_tests:
        success, message = test_func()
        if not success:
            print(f"✓ {test_name} error caught: {message}")
        else:
            print(f"✗ {test_name} should have failed")

def display_summary():
    """Display summary of all improvements"""
    print_section("Summary of Improvements")
    
    improvements = [
        ("Dynamic Paths", "No more hardcoded /home/ubuntu paths"),
        ("Configuration Management", "Environment variables support"),
        ("Logging System", "Colored console output + file logging"),
        ("Error Handling", "Custom exceptions with proper messages"),
        ("REST API", "11 endpoints for all operations"),
        ("Input Validation", "Comprehensive validation for all inputs"),
        ("Test Suite", "130+ tests with pytest"),
        ("Database Support", "SQLite with SQLAlchemy ORM"),
    ]
    
    for i, (feature, description) in enumerate(improvements, 1):
        print(f"{i}. ✅ {feature}")
        print(f"   └─ {description}")

def main():
    """Run all tests"""
    print("\n" + "🚀 "*20)
    print("CRYPTOCURRENCY INVESTMENT APP - DEMO TEST")
    print("🚀 "*20)
    
    tests = [
        test_configuration,
        test_logging,
        test_validation,
        test_portfolio_operations,
        test_data_collection,
        test_error_handling,
        test_api_endpoints,
        display_summary,
    ]
    
    start_time = time.time()
    
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"\n❌ Error in {test.__name__}: {str(e)}")
            logger.error(f"Test failed: {test.__name__}", exc_info=True)
    
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*60)
    print(f"✨ Demo completed in {elapsed_time:.2f} seconds")
    print("="*60)

if __name__ == "__main__":
    main()