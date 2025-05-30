"""
Comprehensive input validation module for the cryptocurrency investment app
"""

import re
from datetime import datetime, timedelta
from typing import Any, Optional, Union, List, Dict
from utils.logger import setup_logger
from utils.error_handlers import ValidationError

logger = setup_logger(__name__)

# Cryptocurrency validation
VALID_CRYPTO_SYMBOLS = ['BTC', 'ETH']
VALID_CRYPTO_PAIRS = ['BTC-USD', 'ETH-USD']

# Numeric constraints
MIN_TRANSACTION_AMOUNT = 0.00000001  # Minimum crypto amount (1 satoshi)
MAX_TRANSACTION_AMOUNT = 1000000  # Maximum crypto amount
MIN_PRICE = 0.01  # Minimum price in CHF
MAX_PRICE = 10000000  # Maximum price in CHF
MIN_INVESTMENT = 1  # Minimum investment in CHF
MAX_INVESTMENT = 100000000  # Maximum investment in CHF

# Date constraints
MIN_DATE = datetime(2010, 1, 1)  # Bitcoin launch
MAX_FUTURE_DAYS = 365  # Maximum days into future for predictions

# String constraints
MIN_STRING_LENGTH = 1
MAX_STRING_LENGTH = 1000
USERNAME_PATTERN = re.compile(r'^[a-zA-Z0-9_-]{3,30}$')
EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')

class Validator:
    """Base validator class"""
    
    @staticmethod
    def validate_required(value: Any, field_name: str) -> Any:
        """Validate that a value is not None or empty"""
        if value is None or (isinstance(value, str) and not value.strip()):
            raise ValidationError(f"{field_name} is required")
        return value
    
    @staticmethod
    def validate_type(value: Any, expected_type: type, field_name: str) -> Any:
        """Validate that a value is of the expected type"""
        if not isinstance(value, expected_type):
            raise ValidationError(f"{field_name} must be of type {expected_type.__name__}")
        return value

class CryptoValidator(Validator):
    """Cryptocurrency-specific validators"""
    
    @staticmethod
    def validate_symbol(symbol: str) -> str:
        """Validate cryptocurrency symbol"""
        if not symbol:
            raise ValidationError("Symbol is required")
        
        symbol = symbol.upper().strip()
        
        if symbol not in VALID_CRYPTO_SYMBOLS:
            raise ValidationError(
                f"Invalid cryptocurrency symbol: {symbol}. "
                f"Valid symbols are: {', '.join(VALID_CRYPTO_SYMBOLS)}"
            )
        
        return symbol
    
    @staticmethod
    def validate_symbol_pair(pair: str) -> str:
        """Validate cryptocurrency trading pair"""
        if not pair:
            raise ValidationError("Trading pair is required")
        
        pair = pair.upper().strip()
        
        if pair not in VALID_CRYPTO_PAIRS:
            raise ValidationError(
                f"Invalid trading pair: {pair}. "
                f"Valid pairs are: {', '.join(VALID_CRYPTO_PAIRS)}"
            )
        
        return pair
    
    @staticmethod
    def validate_amount(amount: Union[float, str], symbol: Optional[str] = None) -> float:
        """Validate cryptocurrency amount"""
        try:
            amount = float(amount)
        except (TypeError, ValueError):
            raise ValidationError("Amount must be a number")
        
        if amount <= 0:
            raise ValidationError("Amount must be positive")
        
        if amount < MIN_TRANSACTION_AMOUNT:
            raise ValidationError(
                f"Amount must be at least {MIN_TRANSACTION_AMOUNT} "
                f"({'1 satoshi' if symbol == 'BTC' else 'minimum unit'})"
            )
        
        if amount > MAX_TRANSACTION_AMOUNT:
            raise ValidationError(f"Amount cannot exceed {MAX_TRANSACTION_AMOUNT}")
        
        return amount
    
    @staticmethod
    def validate_price(price: Union[float, str]) -> float:
        """Validate cryptocurrency price"""
        try:
            price = float(price)
        except (TypeError, ValueError):
            raise ValidationError("Price must be a number")
        
        if price <= 0:
            raise ValidationError("Price must be positive")
        
        if price < MIN_PRICE:
            raise ValidationError(f"Price must be at least CHF {MIN_PRICE}")
        
        if price > MAX_PRICE:
            raise ValidationError(f"Price cannot exceed CHF {MAX_PRICE}")
        
        return price

class FinancialValidator(Validator):
    """Financial and investment validators"""
    
    @staticmethod
    def validate_investment_amount(amount: Union[float, str]) -> float:
        """Validate investment amount in CHF"""
        try:
            amount = float(amount)
        except (TypeError, ValueError):
            raise ValidationError("Investment amount must be a number")
        
        if amount <= 0:
            raise ValidationError("Investment amount must be positive")
        
        if amount < MIN_INVESTMENT:
            raise ValidationError(f"Minimum investment is CHF {MIN_INVESTMENT}")
        
        if amount > MAX_INVESTMENT:
            raise ValidationError(f"Maximum investment is CHF {MAX_INVESTMENT}")
        
        return amount
    
    @staticmethod
    def validate_percentage(value: Union[float, str], field_name: str = "Percentage") -> float:
        """Validate percentage value (0-100)"""
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValidationError(f"{field_name} must be a number")
        
        if not 0 <= value <= 100:
            raise ValidationError(f"{field_name} must be between 0 and 100")
        
        return value
    
    @staticmethod
    def validate_portfolio_allocation(allocations: Dict[str, float]) -> Dict[str, float]:
        """Validate portfolio allocation percentages"""
        if not allocations:
            raise ValidationError("Portfolio allocations are required")
        
        total = sum(allocations.values())
        
        if abs(total - 100) > 0.01:  # Allow small floating point errors
            raise ValidationError(
                f"Portfolio allocations must sum to 100%, got {total:.2f}%"
            )
        
        for symbol, percentage in allocations.items():
            if symbol != 'cash':
                CryptoValidator.validate_symbol(symbol)
            FinancialValidator.validate_percentage(percentage, f"Allocation for {symbol}")
        
        return allocations

class DateValidator(Validator):
    """Date and time validators"""
    
    @staticmethod
    def validate_date(date_value: Union[str, datetime], field_name: str = "Date") -> datetime:
        """Validate date value"""
        if isinstance(date_value, str):
            try:
                date_value = datetime.fromisoformat(date_value.replace('Z', '+00:00'))
            except ValueError:
                try:
                    date_value = datetime.strptime(date_value, '%Y-%m-%d')
                except ValueError:
                    raise ValidationError(f"{field_name} must be in ISO format (YYYY-MM-DD)")
        
        if not isinstance(date_value, datetime):
            raise ValidationError(f"{field_name} must be a valid date")
        
        # Make MIN_DATE timezone-aware if date_value is timezone-aware
        min_date = MIN_DATE
        if date_value.tzinfo is not None and MIN_DATE.tzinfo is None:
            import pytz
            min_date = pytz.UTC.localize(MIN_DATE)
        elif date_value.tzinfo is None and MIN_DATE.tzinfo is not None:
            date_value = date_value.replace(tzinfo=pytz.UTC)
            
        if date_value < min_date:
            raise ValidationError(f"{field_name} cannot be before {MIN_DATE.date()}")
        
        if date_value > datetime.now() + timedelta(days=MAX_FUTURE_DAYS):
            raise ValidationError(f"{field_name} cannot be more than {MAX_FUTURE_DAYS} days in the future")
        
        return date_value
    
    @staticmethod
    def validate_date_range(start_date: Union[str, datetime], 
                          end_date: Union[str, datetime]) -> tuple[datetime, datetime]:
        """Validate date range"""
        start = DateValidator.validate_date(start_date, "Start date")
        end = DateValidator.validate_date(end_date, "End date")
        
        if start > end:
            raise ValidationError("Start date must be before end date")
        
        if (end - start).days > 365 * 10:  # Maximum 10 years
            raise ValidationError("Date range cannot exceed 10 years")
        
        return start, end
    
    @staticmethod
    def validate_days(days: Union[int, str], field_name: str = "Days") -> int:
        """Validate number of days"""
        try:
            days = int(days)
        except (TypeError, ValueError):
            raise ValidationError(f"{field_name} must be an integer")
        
        if days <= 0:
            raise ValidationError(f"{field_name} must be positive")
        
        if days > 365 * 5:  # Maximum 5 years
            raise ValidationError(f"{field_name} cannot exceed {365 * 5} days (5 years)")
        
        return days

class StringValidator(Validator):
    """String validators"""
    
    @staticmethod
    def validate_string_length(value: str, min_length: int = MIN_STRING_LENGTH, 
                             max_length: int = MAX_STRING_LENGTH, 
                             field_name: str = "String") -> str:
        """Validate string length"""
        if not isinstance(value, str):
            raise ValidationError(f"{field_name} must be a string")
        
        value = value.strip()
        
        if len(value) < min_length:
            raise ValidationError(f"{field_name} must be at least {min_length} characters")
        
        if len(value) > max_length:
            raise ValidationError(f"{field_name} cannot exceed {max_length} characters")
        
        return value
    
    @staticmethod
    def validate_username(username: str) -> str:
        """Validate username format"""
        username = username.strip()
        
        if not USERNAME_PATTERN.match(username):
            raise ValidationError(
                "Username must be 3-30 characters long and contain only "
                "letters, numbers, underscores, and hyphens"
            )
        
        return username
    
    @staticmethod
    def validate_email(email: str) -> str:
        """Validate email format"""
        email = email.strip().lower()
        
        if not EMAIL_PATTERN.match(email):
            raise ValidationError("Invalid email format")
        
        return email
    
    @staticmethod
    def sanitize_input(value: str, allow_html: bool = False) -> str:
        """Sanitize user input to prevent XSS"""
        if not isinstance(value, str):
            return str(value)
        
        # Remove null bytes
        value = value.replace('\x00', '')
        
        if not allow_html:
            # Escape HTML special characters
            value = (value
                    .replace('&', '&amp;')
                    .replace('<', '&lt;')
                    .replace('>', '&gt;')
                    .replace('"', '&quot;')
                    .replace("'", '&#x27;'))
        
        return value.strip()

class RequestValidator:
    """HTTP request validators"""
    
    @staticmethod
    def validate_pagination(page: Union[int, str] = 1, 
                          per_page: Union[int, str] = 20) -> tuple[int, int]:
        """Validate pagination parameters"""
        try:
            page = int(page)
            per_page = int(per_page)
        except (TypeError, ValueError):
            raise ValidationError("Page and per_page must be integers")
        
        if page < 1:
            raise ValidationError("Page must be at least 1")
        
        if per_page < 1:
            raise ValidationError("Per page must be at least 1")
        
        if per_page > 100:
            raise ValidationError("Per page cannot exceed 100")
        
        return page, per_page
    
    @staticmethod
    def validate_sort_order(sort_by: str, allowed_fields: List[str], 
                          order: str = 'asc') -> tuple[str, str]:
        """Validate sort parameters"""
        if sort_by not in allowed_fields:
            raise ValidationError(
                f"Invalid sort field: {sort_by}. "
                f"Allowed fields are: {', '.join(allowed_fields)}"
            )
        
        order = order.lower()
        if order not in ['asc', 'desc']:
            raise ValidationError("Sort order must be 'asc' or 'desc'")
        
        return sort_by, order

# Convenience functions
def validate_transaction_request(data: dict) -> dict:
    """Validate a transaction request (buy/sell)"""
    validated = {}
    
    # Validate symbol
    validated['symbol'] = CryptoValidator.validate_symbol(
        Validator.validate_required(data.get('symbol'), 'Symbol')
    )
    
    # Validate amount
    validated['amount'] = CryptoValidator.validate_amount(
        Validator.validate_required(data.get('amount'), 'Amount'),
        validated['symbol']
    )
    
    # Validate price
    validated['price'] = CryptoValidator.validate_price(
        Validator.validate_required(data.get('price'), 'Price')
    )
    
    return validated

def validate_data_request(params: dict) -> dict:
    """Validate data request parameters"""
    validated = {}
    
    # Validate symbol if provided
    if 'symbol' in params:
        validated['symbol'] = CryptoValidator.validate_symbol(params['symbol'])
    
    # Validate days if provided
    if 'days' in params:
        validated['days'] = DateValidator.validate_days(params['days'])
    
    # Validate date range if provided
    if 'start_date' in params and 'end_date' in params:
        validated['start_date'], validated['end_date'] = DateValidator.validate_date_range(
            params['start_date'], params['end_date']
        )
    
    return validated

# Example usage and tests
if __name__ == "__main__":
    # Test crypto validators
    try:
        print("Testing crypto validators...")
        print(f"Valid symbol: {CryptoValidator.validate_symbol('btc')}")
        print(f"Valid amount: {CryptoValidator.validate_amount('0.001')}")
        print(f"Valid price: {CryptoValidator.validate_price('50000')}")
    except ValidationError as e:
        print(f"Validation error: {e}")
    
    # Test transaction validation
    try:
        print("\nTesting transaction validation...")
        transaction = validate_transaction_request({
            'symbol': 'BTC',
            'amount': '0.001',
            'price': '50000'
        })
        print(f"Valid transaction: {transaction}")
    except ValidationError as e:
        print(f"Validation error: {e}")
    
    # Test invalid data
    try:
        print("\nTesting invalid data...")
        CryptoValidator.validate_symbol('INVALID')
    except ValidationError as e:
        print(f"Expected error: {e}")