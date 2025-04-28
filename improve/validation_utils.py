"""
Validation Utilities Module

This module provides utility functions for validating various types of data
using the rules defined in validation_config.py.
"""

import re
from typing import Dict, List, Tuple, Union, Optional
from datetime import datetime
from validation_config import (
    PHONE_VALIDATION,
    AGE_VALIDATION,
    PASSWORD_VALIDATION,
    DATE_VALIDATION,
    EMAIL_VALIDATION,
    USERNAME_VALIDATION,
)

class ValidationError(Exception):
    """Custom exception for validation errors."""
    def __init__(self, message: str, code: str = None):
        self.message = message
        self.code = code
        super().__init__(self.message)

def clean_phone_number(phone: str) -> str:
    """
    Clean and standardize phone number format.
    
    Args:
        phone: Phone number string to clean
        
    Returns:
        Cleaned phone number string
    """
    if not phone:
        return ''
        
    # Apply cleanup patterns
    cleaned = phone
    for pattern in PHONE_VALIDATION['CLEANUP_PATTERNS']:
        cleaned = re.sub(pattern, '', cleaned)
    
    return cleaned

def validate_phone_number(phone: str) -> Tuple[bool, Optional[str]]:
    """
    Validate phone number format and length.
    
    Args:
        phone: Phone number to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not phone:
        return False, PHONE_VALIDATION['ERROR_MESSAGES']['INVALID_FORMAT']
    
    cleaned = clean_phone_number(phone)
    
    # Check length
    if not (PHONE_VALIDATION['MIN_LENGTH'] <= len(cleaned) <= PHONE_VALIDATION['MAX_LENGTH']):
        return False, PHONE_VALIDATION['ERROR_MESSAGES']['INVALID_LENGTH']
    
    # Check against patterns
    for pattern in PHONE_VALIDATION['PATTERNS']:
        if re.match(pattern, cleaned):
            return True, None
    
    return False, PHONE_VALIDATION['ERROR_MESSAGES']['INVALID_FORMAT']

def validate_age(age: Union[str, int]) -> Tuple[bool, Optional[str]]:
    """
    Validate age value and range.
    
    Args:
        age: Age value to validate (string or integer)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not age:
        return False, AGE_VALIDATION['ERROR_MESSAGES']['INVALID_FORMAT']
    
    # Convert to string for pattern matching
    age_str = str(age)
    
    # Check against patterns
    for pattern in AGE_VALIDATION['PATTERNS']:
        if re.match(pattern, age_str):
            # Extract numeric value
            numeric_age = int(''.join(filter(str.isdigit, age_str)))
            if AGE_VALIDATION['MIN_AGE'] <= numeric_age <= AGE_VALIDATION['MAX_AGE']:
                return True, None
    
    return False, AGE_VALIDATION['ERROR_MESSAGES']['INVALID_FORMAT']

def validate_password(password: str) -> Tuple[bool, List[str]]:
    """
    Validate password against requirements.
    
    Args:
        password: Password to validate
        
    Returns:
        Tuple of (is_valid, list_of_error_messages)
    """
    errors = []
    
    # Check length
    if len(password) < PASSWORD_VALIDATION['MIN_LENGTH']:
        errors.append(PASSWORD_VALIDATION['ERROR_MESSAGES']['TOO_SHORT'])
    elif len(password) > PASSWORD_VALIDATION['MAX_LENGTH']:
        errors.append(PASSWORD_VALIDATION['ERROR_MESSAGES']['TOO_LONG'])
    
    # Check requirements
    for req, required in PASSWORD_VALIDATION['REQUIREMENTS'].items():
        if required and not re.search(PASSWORD_VALIDATION['PATTERNS'][req], password):
            errors.append(PASSWORD_VALIDATION['ERROR_MESSAGES'][f'MISSING_{req}'])
    
    return len(errors) == 0, errors

def validate_date(date_str: str, format: str = 'ISO') -> Tuple[bool, Optional[str]]:
    """
    Validate date string format.
    
    Args:
        date_str: Date string to validate
        format: Expected date format ('ISO', 'US', 'EU', 'SHORT_YEAR')
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not date_str:
        return False, DATE_VALIDATION['ERROR_MESSAGES']['INVALID_FORMAT']
    
    pattern = DATE_VALIDATION['PATTERNS'].get(format)
    if not pattern:
        return False, DATE_VALIDATION['ERROR_MESSAGES']['INVALID_FORMAT']
    
    if not re.match(pattern, date_str):
        return False, DATE_VALIDATION['ERROR_MESSAGES']['INVALID_FORMAT']
    
    try:
        # Try parsing the date to ensure it's valid
        if format == 'ISO':
            datetime.strptime(date_str, '%Y-%m-%d')
        elif format in ['US', 'EU']:
            datetime.strptime(date_str, '%m/%d/%Y')
        elif format == 'SHORT_YEAR':
            datetime.strptime(date_str, '%m/%y')
    except ValueError:
        return False, DATE_VALIDATION['ERROR_MESSAGES']['INVALID_RANGE']
    
    return True, None

def validate_email(email: str) -> Tuple[bool, Optional[str]]:
    """
    Validate email address format and length.
    
    Args:
        email: Email address to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not email:
        return False, EMAIL_VALIDATION['ERROR_MESSAGES']['INVALID_FORMAT']
    
    if len(email) > EMAIL_VALIDATION['MAX_LENGTH']:
        return False, EMAIL_VALIDATION['ERROR_MESSAGES']['TOO_LONG']
    
    if not re.match(EMAIL_VALIDATION['PATTERNS']['STRICT'], email):
        return False, EMAIL_VALIDATION['ERROR_MESSAGES']['INVALID_FORMAT']
    
    return True, None

def validate_username(username: str) -> Tuple[bool, Optional[str]]:
    """
    Validate username format and length.
    
    Args:
        username: Username to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not username:
        return False, USERNAME_VALIDATION['ERROR_MESSAGES']['INVALID_FORMAT']
    
    if len(username) < USERNAME_VALIDATION['MIN_LENGTH']:
        return False, USERNAME_VALIDATION['ERROR_MESSAGES']['TOO_SHORT']
    
    if len(username) > USERNAME_VALIDATION['MAX_LENGTH']:
        return False, USERNAME_VALIDATION['ERROR_MESSAGES']['TOO_LONG']
    
    if not re.match(USERNAME_VALIDATION['PATTERN'], username):
        return False, USERNAME_VALIDATION['ERROR_MESSAGES']['INVALID_FORMAT']
    
    return True, None 