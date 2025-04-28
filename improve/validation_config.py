"""
Validation Configuration Module

This module contains all validation rules and patterns used throughout the application.
It provides a centralized place to manage and update validation requirements.
"""

from datetime import timedelta
from typing import Dict, List, Union

# Cache Configuration
CACHE_CONFIG = {
    'DEFAULT_DURATION': timedelta(minutes=30),
    'WEATHER_CACHE_DURATION': timedelta(minutes=30),
    'LOCATION_CACHE_DURATION': timedelta(minutes=30),
    'USER_SESSION_DURATION': timedelta(hours=24),
}

# Phone Number Validation
PHONE_VALIDATION = {
    'MIN_LENGTH': 10,
    'MAX_LENGTH': 15,
    'PATTERNS': [
        # International format with country code
        r'^\+?[1-9]\d{1,14}$',
        # Indian format
        r'^[6-9]\d{9}$',
        # Generic format with separators
        r'^\+?[\d\s\-\(\)]{10,15}$'
    ],
    'CLEANUP_PATTERNS': [
        r'[\s\-\(\)]',  # Remove spaces, hyphens, and parentheses
        r'^\+?',        # Remove leading plus
    ],
    'ERROR_MESSAGES': {
        'INVALID_FORMAT': 'Please enter a valid phone number',
        'INVALID_LENGTH': 'Phone number must be between 10 and 15 digits',
        'INVALID_COUNTRY_CODE': 'Please include a valid country code',
    }
}

# Age Validation
AGE_VALIDATION = {
    'MIN_AGE': 0,
    'MAX_AGE': 150,
    'PATTERNS': [
        r'^\d{1,3}$',  # Simple numeric age
        r'^\d{1,3}\s*(?:years?|yrs?)?$',  # Age with optional unit
    ],
    'ERROR_MESSAGES': {
        'INVALID_FORMAT': 'Please enter a valid age',
        'OUT_OF_RANGE': 'Age must be between 0 and 150 years',
    }
}

# Password Validation
PASSWORD_VALIDATION = {
    'MIN_LENGTH': 8,
    'MAX_LENGTH': 128,
    'REQUIREMENTS': {
        'UPPERCASE': True,
        'LOWERCASE': True,
        'NUMBERS': True,
        'SPECIAL_CHARS': True,
    },
    'PATTERNS': {
        'UPPERCASE': r'[A-Z]',
        'LOWERCASE': r'[a-z]',
        'NUMBERS': r'\d',
        'SPECIAL_CHARS': r'[!@#$%^&*(),.?":{}|<>]',
    },
    'ERROR_MESSAGES': {
        'TOO_SHORT': 'Password must be at least 8 characters long',
        'TOO_LONG': 'Password must not exceed 128 characters',
        'MISSING_UPPERCASE': 'Password must include at least one uppercase letter',
        'MISSING_LOWERCASE': 'Password must include at least one lowercase letter',
        'MISSING_NUMBER': 'Password must include at least one number',
        'MISSING_SPECIAL': 'Password must include at least one special character',
    }
}

# Date Validation
DATE_VALIDATION = {
    'PATTERNS': {
        'ISO': r'^\d{4}-\d{2}-\d{2}$',
        'US': r'^(?:0[1-9]|1[0-2])/(?:0[1-9]|[12]\d|3[01])/\d{4}$',
        'EU': r'^(?:0[1-9]|[12]\d|3[01])/(?:0[1-9]|1[0-2])/\d{4}$',
        'SHORT_YEAR': r'^(?:0[1-9]|1[0-2])/\d{2}$',
    },
    'ERROR_MESSAGES': {
        'INVALID_FORMAT': 'Please enter a valid date',
        'INVALID_RANGE': 'Date is out of valid range',
    }
}

# Email Validation
EMAIL_VALIDATION = {
    'MAX_LENGTH': 254,
    'PATTERNS': {
        'BASIC': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
        'STRICT': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
    },
    'ERROR_MESSAGES': {
        'INVALID_FORMAT': 'Please enter a valid email address',
        'TOO_LONG': 'Email address is too long',
    }
}

# Username Validation
USERNAME_VALIDATION = {
    'MIN_LENGTH': 3,
    'MAX_LENGTH': 16,
    'PATTERN': r'^[a-zA-Z0-9_-]{3,16}$',
    'ERROR_MESSAGES': {
        'INVALID_FORMAT': 'Username must be 3-16 characters long and can only contain letters, numbers, underscores, and hyphens',
        'TOO_SHORT': 'Username must be at least 3 characters long',
        'TOO_LONG': 'Username must not exceed 16 characters',
    }
}

def get_validation_rules(rule_type: str) -> Dict:
    """
    Get validation rules for a specific type.
    
    Args:
        rule_type: Type of validation rule to retrieve
        
    Returns:
        Dictionary containing validation rules
    """
    validation_map = {
        'phone': PHONE_VALIDATION,
        'age': AGE_VALIDATION,
        'password': PASSWORD_VALIDATION,
        'date': DATE_VALIDATION,
        'email': EMAIL_VALIDATION,
        'username': USERNAME_VALIDATION,
        'cache': CACHE_CONFIG,
    }
    return validation_map.get(rule_type, {}) 