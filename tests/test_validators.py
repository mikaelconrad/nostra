"""
Tests for the validators module
"""

import pytest
from datetime import datetime, timedelta
from utils.validators import (
    CryptoValidator, FinancialValidator, DateValidator, 
    StringValidator, RequestValidator, validate_transaction_request
)
from utils.error_handlers import ValidationError

class TestCryptoValidator:
    """Test suite for cryptocurrency validators"""
    
    def test_validate_symbol_valid(self):
        """Test valid cryptocurrency symbols"""
        assert CryptoValidator.validate_symbol('BTC') == 'BTC'
        assert CryptoValidator.validate_symbol('eth') == 'ETH'  # Case insensitive
        assert CryptoValidator.validate_symbol(' XRP ') == 'XRP'  # Strips whitespace
    
    def test_validate_symbol_invalid(self):
        """Test invalid cryptocurrency symbols"""
        with pytest.raises(ValidationError) as exc:
            CryptoValidator.validate_symbol('DOGE')
        assert 'Invalid cryptocurrency symbol' in str(exc.value)
        
        with pytest.raises(ValidationError):
            CryptoValidator.validate_symbol('')
        
        with pytest.raises(ValidationError):
            CryptoValidator.validate_symbol(None)
    
    def test_validate_amount_valid(self):
        """Test valid cryptocurrency amounts"""
        assert CryptoValidator.validate_amount(0.001) == 0.001
        assert CryptoValidator.validate_amount('0.5') == 0.5
        assert CryptoValidator.validate_amount(100) == 100.0
    
    def test_validate_amount_invalid(self):
        """Test invalid cryptocurrency amounts"""
        # Negative amount
        with pytest.raises(ValidationError) as exc:
            CryptoValidator.validate_amount(-0.001)
        assert 'must be positive' in str(exc.value)
        
        # Too small
        with pytest.raises(ValidationError):
            CryptoValidator.validate_amount(0.000000001)
        
        # Too large
        with pytest.raises(ValidationError):
            CryptoValidator.validate_amount(10000000)
        
        # Invalid type
        with pytest.raises(ValidationError):
            CryptoValidator.validate_amount('invalid')
    
    def test_validate_price_valid(self):
        """Test valid cryptocurrency prices"""
        assert CryptoValidator.validate_price(50000) == 50000.0
        assert CryptoValidator.validate_price('2500.50') == 2500.50
        assert CryptoValidator.validate_price(0.5) == 0.5
    
    def test_validate_price_invalid(self):
        """Test invalid cryptocurrency prices"""
        # Too low
        with pytest.raises(ValidationError):
            CryptoValidator.validate_price(0.001)
        
        # Too high
        with pytest.raises(ValidationError):
            CryptoValidator.validate_price(100000000)
        
        # Negative
        with pytest.raises(ValidationError):
            CryptoValidator.validate_price(-100)

class TestFinancialValidator:
    """Test suite for financial validators"""
    
    def test_validate_investment_amount_valid(self):
        """Test valid investment amounts"""
        assert FinancialValidator.validate_investment_amount(100) == 100.0
        assert FinancialValidator.validate_investment_amount('5000') == 5000.0
    
    def test_validate_investment_amount_invalid(self):
        """Test invalid investment amounts"""
        # Too small
        with pytest.raises(ValidationError):
            FinancialValidator.validate_investment_amount(0.5)
        
        # Too large
        with pytest.raises(ValidationError):
            FinancialValidator.validate_investment_amount(200000000)
        
        # Negative
        with pytest.raises(ValidationError):
            FinancialValidator.validate_investment_amount(-100)
    
    def test_validate_percentage_valid(self):
        """Test valid percentage values"""
        assert FinancialValidator.validate_percentage(50) == 50.0
        assert FinancialValidator.validate_percentage('25.5') == 25.5
        assert FinancialValidator.validate_percentage(0) == 0.0
        assert FinancialValidator.validate_percentage(100) == 100.0
    
    def test_validate_percentage_invalid(self):
        """Test invalid percentage values"""
        with pytest.raises(ValidationError):
            FinancialValidator.validate_percentage(-5)
        
        with pytest.raises(ValidationError):
            FinancialValidator.validate_percentage(101)
        
        with pytest.raises(ValidationError):
            FinancialValidator.validate_percentage('invalid')
    
    def test_validate_portfolio_allocation_valid(self):
        """Test valid portfolio allocations"""
        allocations = {
            'BTC': 40.0,
            'ETH': 35.0,
            'cash': 25.0
        }
        result = FinancialValidator.validate_portfolio_allocation(allocations)
        assert result == allocations
    
    def test_validate_portfolio_allocation_invalid(self):
        """Test invalid portfolio allocations"""
        # Doesn't sum to 100
        with pytest.raises(ValidationError) as exc:
            FinancialValidator.validate_portfolio_allocation({
                'BTC': 40.0,
                'ETH': 30.0,
                'cash': 20.0
            })
        assert 'must sum to 100%' in str(exc.value)
        
        # Invalid symbol
        with pytest.raises(ValidationError):
            FinancialValidator.validate_portfolio_allocation({
                'DOGE': 50.0,
                'cash': 50.0
            })
        
        # Negative allocation
        with pytest.raises(ValidationError):
            FinancialValidator.validate_portfolio_allocation({
                'BTC': 120.0,
                'cash': -20.0
            })

class TestDateValidator:
    """Test suite for date validators"""
    
    def test_validate_date_valid(self):
        """Test valid date values"""
        # String dates
        result = DateValidator.validate_date('2024-01-15')
        assert isinstance(result, datetime)
        
        # ISO format with timezone
        result = DateValidator.validate_date('2024-01-15T10:30:00+00:00')
        assert isinstance(result, datetime)
        
        # Datetime object
        now = datetime.now()
        result = DateValidator.validate_date(now)
        assert result == now
    
    def test_validate_date_invalid(self):
        """Test invalid date values"""
        # Before minimum date
        with pytest.raises(ValidationError):
            DateValidator.validate_date('2009-01-01')
        
        # Too far in future
        future_date = datetime.now() + timedelta(days=400)
        with pytest.raises(ValidationError):
            DateValidator.validate_date(future_date)
        
        # Invalid format
        with pytest.raises(ValidationError):
            DateValidator.validate_date('15/01/2024')
    
    def test_validate_date_range_valid(self):
        """Test valid date ranges"""
        start = '2024-01-01'
        end = '2024-12-31'
        start_dt, end_dt = DateValidator.validate_date_range(start, end)
        assert start_dt < end_dt
    
    def test_validate_date_range_invalid(self):
        """Test invalid date ranges"""
        # Start after end
        with pytest.raises(ValidationError):
            DateValidator.validate_date_range('2024-12-31', '2024-01-01')
        
        # Range too large
        with pytest.raises(ValidationError):
            DateValidator.validate_date_range('2010-01-01', '2025-01-01')
    
    def test_validate_days_valid(self):
        """Test valid days values"""
        assert DateValidator.validate_days(30) == 30
        assert DateValidator.validate_days('365') == 365
    
    def test_validate_days_invalid(self):
        """Test invalid days values"""
        with pytest.raises(ValidationError):
            DateValidator.validate_days(0)
        
        with pytest.raises(ValidationError):
            DateValidator.validate_days(-10)
        
        with pytest.raises(ValidationError):
            DateValidator.validate_days(2000)  # More than 5 years

class TestStringValidator:
    """Test suite for string validators"""
    
    def test_validate_string_length_valid(self):
        """Test valid string lengths"""
        result = StringValidator.validate_string_length('hello', 1, 10)
        assert result == 'hello'
        
        # Strips whitespace
        result = StringValidator.validate_string_length('  test  ', 1, 10)
        assert result == 'test'
    
    def test_validate_string_length_invalid(self):
        """Test invalid string lengths"""
        # Too short
        with pytest.raises(ValidationError):
            StringValidator.validate_string_length('', 1, 10)
        
        # Too long
        with pytest.raises(ValidationError):
            StringValidator.validate_string_length('a' * 1001, 1, 1000)
        
        # Not a string
        with pytest.raises(ValidationError):
            StringValidator.validate_string_length(123, 1, 10)
    
    def test_validate_username_valid(self):
        """Test valid usernames"""
        assert StringValidator.validate_username('john_doe') == 'john_doe'
        assert StringValidator.validate_username('user123') == 'user123'
        assert StringValidator.validate_username('test-user') == 'test-user'
    
    def test_validate_username_invalid(self):
        """Test invalid usernames"""
        # Too short
        with pytest.raises(ValidationError):
            StringValidator.validate_username('ab')
        
        # Invalid characters
        with pytest.raises(ValidationError):
            StringValidator.validate_username('user@123')
        
        # Too long
        with pytest.raises(ValidationError):
            StringValidator.validate_username('a' * 31)
    
    def test_validate_email_valid(self):
        """Test valid email addresses"""
        assert StringValidator.validate_email('test@example.com') == 'test@example.com'
        assert StringValidator.validate_email('USER@EXAMPLE.COM') == 'user@example.com'
        assert StringValidator.validate_email('test.user+tag@example.co.uk') == 'test.user+tag@example.co.uk'
    
    def test_validate_email_invalid(self):
        """Test invalid email addresses"""
        invalid_emails = [
            'invalid',
            '@example.com',
            'user@',
            'user@@example.com',
            'user@example',
            'user @example.com'
        ]
        
        for email in invalid_emails:
            with pytest.raises(ValidationError):
                StringValidator.validate_email(email)
    
    def test_sanitize_input(self):
        """Test input sanitization"""
        # HTML escaping
        assert StringValidator.sanitize_input('<script>alert("xss")</script>') == '&lt;script&gt;alert(&quot;xss&quot;)&lt;/script&gt;'
        assert StringValidator.sanitize_input("O'Reilly") == "O&#x27;Reilly"
        
        # Null byte removal
        assert StringValidator.sanitize_input('test\x00string') == 'teststring'
        
        # Allow HTML when specified
        assert StringValidator.sanitize_input('<b>bold</b>', allow_html=True) == '<b>bold</b>'

class TestRequestValidator:
    """Test suite for request validators"""
    
    def test_validate_pagination_valid(self):
        """Test valid pagination parameters"""
        page, per_page = RequestValidator.validate_pagination(2, 50)
        assert page == 2
        assert per_page == 50
        
        # String inputs
        page, per_page = RequestValidator.validate_pagination('3', '20')
        assert page == 3
        assert per_page == 20
    
    def test_validate_pagination_invalid(self):
        """Test invalid pagination parameters"""
        # Page < 1
        with pytest.raises(ValidationError):
            RequestValidator.validate_pagination(0, 20)
        
        # Per page > 100
        with pytest.raises(ValidationError):
            RequestValidator.validate_pagination(1, 101)
        
        # Invalid types
        with pytest.raises(ValidationError):
            RequestValidator.validate_pagination('invalid', 20)
    
    def test_validate_sort_order_valid(self):
        """Test valid sort parameters"""
        allowed = ['date', 'price', 'amount']
        
        field, order = RequestValidator.validate_sort_order('date', allowed)
        assert field == 'date'
        assert order == 'asc'
        
        field, order = RequestValidator.validate_sort_order('price', allowed, 'desc')
        assert field == 'price'
        assert order == 'desc'
    
    def test_validate_sort_order_invalid(self):
        """Test invalid sort parameters"""
        allowed = ['date', 'price', 'amount']
        
        # Invalid field
        with pytest.raises(ValidationError):
            RequestValidator.validate_sort_order('invalid', allowed)
        
        # Invalid order
        with pytest.raises(ValidationError):
            RequestValidator.validate_sort_order('date', allowed, 'invalid')

class TestTransactionValidation:
    """Test suite for transaction request validation"""
    
    def test_validate_transaction_request_valid(self):
        """Test valid transaction request"""
        data = {
            'symbol': 'BTC',
            'amount': '0.1',
            'price': '50000'
        }
        
        result = validate_transaction_request(data)
        
        assert result['symbol'] == 'BTC'
        assert result['amount'] == 0.1
        assert result['price'] == 50000.0
    
    def test_validate_transaction_request_missing_fields(self):
        """Test transaction request with missing fields"""
        # Missing symbol
        with pytest.raises(ValidationError) as exc:
            validate_transaction_request({'amount': '0.1', 'price': '50000'})
        assert 'Symbol is required' in str(exc.value)
        
        # Missing amount
        with pytest.raises(ValidationError) as exc:
            validate_transaction_request({'symbol': 'BTC', 'price': '50000'})
        assert 'Amount is required' in str(exc.value)
        
        # Missing price
        with pytest.raises(ValidationError) as exc:
            validate_transaction_request({'symbol': 'BTC', 'amount': '0.1'})
        assert 'Price is required' in str(exc.value)
    
    def test_validate_transaction_request_invalid_values(self):
        """Test transaction request with invalid values"""
        # Invalid symbol
        with pytest.raises(ValidationError):
            validate_transaction_request({
                'symbol': 'INVALID',
                'amount': '0.1',
                'price': '50000'
            })
        
        # Negative amount
        with pytest.raises(ValidationError):
            validate_transaction_request({
                'symbol': 'BTC',
                'amount': '-0.1',
                'price': '50000'
            })
        
        # Zero price
        with pytest.raises(ValidationError):
            validate_transaction_request({
                'symbol': 'BTC',
                'amount': '0.1',
                'price': '0'
            })