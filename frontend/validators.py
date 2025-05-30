"""
Frontend validation utilities for the cryptocurrency investment app
"""

from typing import Optional, Union

class FrontendValidator:
    """Client-side validation utilities"""
    
    @staticmethod
    def validate_transaction_form(symbol: Optional[str], 
                                amount: Optional[str], 
                                price: Optional[str]) -> tuple[bool, str]:
        """
        Validate transaction form inputs
        Returns: (is_valid, error_message)
        """
        # Check required fields
        if not symbol:
            return False, "Please select a cryptocurrency"
        
        if not amount:
            return False, "Please enter an amount"
        
        if not price:
            return False, "Please enter a price"
        
        # Validate amount
        try:
            amount_float = float(amount)
            if amount_float <= 0:
                return False, "Amount must be greater than 0"
            if amount_float < 0.00000001:
                return False, "Amount is too small (minimum: 0.00000001)"
            if amount_float > 1000000:
                return False, "Amount is too large (maximum: 1,000,000)"
        except ValueError:
            return False, "Amount must be a valid number"
        
        # Validate price
        try:
            price_float = float(price)
            if price_float <= 0:
                return False, "Price must be greater than 0"
            if price_float < 0.01:
                return False, "Price is too low (minimum: CHF 0.01)"
            if price_float > 10000000:
                return False, "Price is too high (maximum: CHF 10,000,000)"
        except ValueError:
            return False, "Price must be a valid number"
        
        # Calculate total cost
        try:
            total = float(amount) * float(price)
            if total < 0.01:
                return False, "Transaction value too small (minimum: CHF 0.01)"
        except:
            return False, "Invalid transaction values"
        
        return True, ""
    
    @staticmethod
    def format_number(value: Union[str, float], decimals: int = 8) -> str:
        """Format number for display"""
        try:
            num = float(value)
            if decimals == 0:
                return f"{num:,.0f}"
            else:
                return f"{num:,.{decimals}f}".rstrip('0').rstrip('.')
        except:
            return str(value)
    
    @staticmethod
    def format_currency(value: Union[str, float], currency: str = "CHF") -> str:
        """Format currency for display"""
        try:
            num = float(value)
            return f"{currency} {num:,.2f}"
        except:
            return f"{currency} {value}"
    
    @staticmethod
    def validate_percentage(value: Optional[str]) -> tuple[bool, str]:
        """Validate percentage input (0-100)"""
        if not value:
            return False, "Percentage is required"
        
        try:
            pct = float(value)
            if pct < 0:
                return False, "Percentage cannot be negative"
            if pct > 100:
                return False, "Percentage cannot exceed 100%"
            return True, ""
        except ValueError:
            return False, "Percentage must be a valid number"
    
    @staticmethod
    def validate_date_range(start_date: Optional[str], 
                          end_date: Optional[str]) -> tuple[bool, str]:
        """Validate date range inputs"""
        if not start_date:
            return False, "Start date is required"
        
        if not end_date:
            return False, "End date is required"
        
        # Simple date comparison (assuming ISO format)
        if start_date > end_date:
            return False, "Start date must be before end date"
        
        return True, ""