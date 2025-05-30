"""
Custom error handlers and exceptions for the cryptocurrency investment app
"""

from utils.logger import setup_logger

logger = setup_logger(__name__)

class CryptoAppError(Exception):
    """Base exception for the crypto investment app"""
    pass

class DataCollectionError(CryptoAppError):
    """Raised when data collection fails"""
    pass

class DataProcessingError(CryptoAppError):
    """Raised when data processing fails"""
    pass

class ModelTrainingError(CryptoAppError):
    """Raised when model training fails"""
    pass

class PortfolioError(CryptoAppError):
    """Raised when portfolio operations fail"""
    pass

class ValidationError(CryptoAppError):
    """Raised when input validation fails"""
    pass

class APIError(CryptoAppError):
    """Raised when API operations fail"""
    pass

class DatabaseError(CryptoAppError):
    """Raised when database operations fail"""
    pass

def handle_error(error_type, message, original_exception=None):
    """
    Centralized error handling function
    
    Args:
        error_type: Type of CryptoAppError to raise
        message: Error message
        original_exception: Original exception that caused this error
    """
    if original_exception:
        logger.error(f"{message}: {str(original_exception)}", exc_info=True)
    else:
        logger.error(message)
    
    raise error_type(message) from original_exception

def safe_execute(func, error_type=CryptoAppError, error_message=None):
    """
    Decorator to safely execute functions with error handling
    
    Args:
        func: Function to execute
        error_type: Type of error to raise on failure
        error_message: Custom error message (optional)
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            message = error_message or f"Error in {func.__name__}"
            handle_error(error_type, message, e)
    return wrapper

def validate_input(value, validator, error_message):
    """
    Validate input with custom validator function
    
    Args:
        value: Value to validate
        validator: Function that returns True if valid
        error_message: Error message if validation fails
    """
    if not validator(value):
        raise ValidationError(error_message)
    return value

# Common validators
def is_positive_number(value):
    """Check if value is a positive number"""
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False

def is_valid_symbol(symbol):
    """Check if cryptocurrency symbol is valid"""
    valid_symbols = ['BTC', 'ETH']
    return symbol in valid_symbols

def is_valid_percentage(value):
    """Check if value is a valid percentage (0-100)"""
    try:
        num = float(value)
        return 0 <= num <= 100
    except (TypeError, ValueError):
        return False

# Example usage
if __name__ == "__main__":
    # Test error handling
    try:
        handle_error(DataCollectionError, "Failed to fetch data", Exception("Network error"))
    except DataCollectionError as e:
        print(f"Caught error: {e}")
    
    # Test validation
    try:
        validate_input(-10, is_positive_number, "Value must be positive")
    except ValidationError as e:
        print(f"Validation error: {e}")
    
    # Test valid input
    result = validate_input(50, is_valid_percentage, "Invalid percentage")
    print(f"Valid percentage: {result}")