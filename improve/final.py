from flask import Flask, render_template, request, redirect, url_for, jsonify, session, flash
import os
import re
import urllib.parse
import urllib.request
import json
import pandas as pd
import io
import os
import time
import requests
import datetime
from pathlib import Path
from pymongo import MongoClient, UpdateOne, TEXT
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling.datamodel.settings import settings
from docling.document_converter import DocumentConverter, PdfFormatOption, WordFormatOption
from docling.pipeline.simple_pipeline import SimplePipeline
import sys
import logging
from werkzeug.utils import secure_filename
import os
import logging.handlers
import secrets  # For secure key generation
from bson import json_util  # Add this import at the top
from fuzzywuzzy import fuzz
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import uuid
import math
from validation_utils import (
    validate_phone_number,
    validate_age,
    validate_password,
    validate_email,
    validate_username,
    ValidationError
)
from io import BytesIO
from PIL import Image

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.handlers.RotatingFileHandler(
            'app.log',
            maxBytes=1024 * 1024,  # 1MB
            backupCount=5,
            encoding='utf-8'
        )
    ]
)
logger = logging.getLogger(__name__)

# Add file handler for logging
file_handler = logging.handlers.RotatingFileHandler(
    'app.log',
    maxBytes=1024 * 1024,  # 1MB
    backupCount=5,
    encoding='utf-8'
)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
))
logger.addHandler(file_handler)

# Set Werkzeug logger to INFO level
logging.getLogger('werkzeug').setLevel(logging.INFO)

# Set docling logger to DEBUG level for more detailed OCR information
logging.getLogger('docling').setLevel(logging.DEBUG)


def load_validation_data(field_type=None):
    """
    Loads validation data from the database with caching.
    
    Args:
        field_type (str, optional): Specific field type to load. If None, loads all.
        
    Returns:
        dict: Dictionary of validation data by field type
    """
    global validation_cache
    
    try:
        # Return cached data if available
        if validation_cache and field_type and field_type in validation_cache:
            return validation_cache[field_type]
        
        if validation_cache and not field_type:
            return validation_cache
        
        # Load data from database
        if field_type:
            data = collection_store_validation.find_one({"standard_variable": field_type})
            if data:
                validation_cache[field_type] = data
                return data
            return None
        else:
            # Load all validation data
            all_data = list(collection_store_validation.find())
            validation_cache = {item.get('standard_variable'): item for item in all_data if item.get('standard_variable')}
            return validation_cache
    except Exception as e:
        logger.error(f"Error loading validation data: {e}")
        return {} if field_type else {}

# Function to get field aliases from validation collections
def get_field_aliases(field_type):
    """
    Get aliases for a specific field type from the validation collections.
    
    Args:
        field_type (str): The type of field to get aliases for (e.g., 'medicine_name', 'doctor_name')
        
    Returns:
        list: List of aliases for the field type
    """
    try:
        # Get validation data from cache or database
        validation_data = load_validation_data(field_type)
        
        if not validation_data:
            logger.warning(f"No validation data found for field type: {field_type}")
            return []
        
        # Extract aliases from the validation data
        aliases = validation_data.get("aliases", [])
        return aliases
        
    except Exception as e:
        logger.error(f"Error getting aliases for field type {field_type}: {e}")
        return []


# Generate a secure secret key
def generate_secret_key():
    return secrets.token_hex(32)  # 32 bytes = 256 bits

# Add Flask app initialization after your existing imports
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.abspath(os.path.join(os.path.dirname(__file__), 'uploads'))
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Load secret key from file or generate a new one
SECRET_KEY_FILE = 'secret_key.txt'
try:
    if os.path.exists(SECRET_KEY_FILE):
        with open(SECRET_KEY_FILE, 'r') as f:
            secret_key = f.read().strip()
    else:
        secret_key = generate_secret_key()
        with open(SECRET_KEY_FILE, 'w') as f:
            f.write(secret_key)
except Exception as e:
    logger.warning(f"Could not load/save secret key file: {e}")
    secret_key = generate_secret_key()  # Fallback to generating new key

app.config['SECRET_KEY'] = secret_key
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'csv'}
# Add this near the top of your file with other global variables
validation_cache = {}
# Ensure upload directory exists with proper permissions
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'], 'markdown'), exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# MongoDB Connection
# At the top of the file, after imports
def connect_to_mongodb():
    """
    Establish MongoDB connection with error handling.
    
    This function creates a connection to the MongoDB server with optimized
    connection pool settings and proper error handling.
    
    Returns:
        MongoClient: MongoDB client instance
        
    Raises:
        Exception: If connection to MongoDB fails
        
    Connection settings:
        - serverSelectionTimeoutMS: 5000ms timeout for server selection
        - maxPoolSize: 50 connections maximum in the pool
        - minPoolSize: 10 connections minimum to maintain in the pool
    """
    try:
        client = MongoClient("mongodb://localhost:47016/",
                            serverSelectionTimeoutMS=5000,
                            maxPoolSize=50,
                            minPoolSize=10)
        # Test the connection
        client.server_info()
        return client
    except Exception as e:
        print(f"Failed to connect to MongoDB: {e}")
        raise

# Replace existing client connection with:
try:
    client = connect_to_mongodb()
    # Database definitions
    db_ocr = client["database_2"]
    db_master = client["database_1"]
    db_metadata = client["database_3"]
    db4 = client["database_4"]
    
    # Collections
    collection_ocr = db_ocr["ocr_data"]
    collection_purchase = db_ocr["purchase_data"]
    collection_metadata = db_metadata["metadata"]  # Metadata for OCR data
    collection_medicine = db_master["master_medicine"]
    collection_weather = db_master["master_weather"]
    collection_doctor = db_master["master_doctor"]
    collection_pharmacy = db_master["master_pharmacy"]
    collection_hospital = db_master["master_hospital"]
    collection_lab = db_master["master_lab"]
    collection_enduser = db_master["master_enduser"]  # Collection for user data
    collection_store_validation = db4["store_detail_validation"]
    collection_table_validation = db4["table_validation"]
    collection_patient = db_master["master_patient"]
except Exception as e:
    print(f"Error setting up database connections: {e}")
    sys.exit(1)

def setup_database_indices():
    """
    Create indices to speed up database operations.
    
    This function sets up MongoDB indices for all collections to optimize query performance.
    It also initializes the metadata collection and sets up schema validation for user data.
    
    Performance considerations:
        - Creates text indices for full-text search capabilities
        - Sets up compound indices for frequently queried field combinations
        - Implements partial indices to avoid indexing null values
        - Adds schema validation to ensure data integrity
        
    Raises:
        Exception: If there's an error setting up indices (except for duplicate key errors)
    """
    try:
        # First, clean up any existing documents with null emails or usernames
        collection_enduser.delete_many({"$or": [{"email": None}, {"username": None}]})
        
        # Create indices with proper validation
        collection_medicine.create_index([("product_name", TEXT)])
        collection_doctor.create_index([("doctor_name", 1)])
        collection_doctor.create_index([("city", 1)])
        collection_doctor.create_index([("state", 1)])
        collection_pharmacy.create_index("pharmacy_name")
        collection_hospital.create_index("hospital_name")
        collection_lab.create_index("lab_name")
        collection_weather.create_index("City")
        collection_ocr.create_index("timestamp")
        collection_purchase.create_index("timestamp")
        collection_metadata.create_index("collection_name", unique=True)
        
        # Initialize metadata collection if it doesn't exist
        metadata_doc = collection_metadata.find_one({"collection_name": "ocr_data"})
        if not metadata_doc:
            initial_metadata = {
                "collection_name": "ocr_data",
                "total_documents": 0,
                "ocr_document_ids": [],
                "created_at": datetime.datetime.now(),
                "last_updated": datetime.datetime.now(),
                "markdown_files": [],
                "status": "active"
            }
            collection_metadata.insert_one(initial_metadata)
            logger.info("Initialized metadata collection for OCR data")
        
        # Create unique indices for both email and username
        collection_enduser.create_index(
            [("email", 1)],
            unique=True,
            partialFilterExpression={"email": {"$type": "string"}}
        )
        collection_enduser.create_index(
            [("username", 1)],
            unique=True,
            partialFilterExpression={"username": {"$type": "string"}}
        )
        
        # Add schema validation for the enduser collection
        db_master.command({
            'collMod': 'master_enduser',
            'validator': {
                '$jsonSchema': {
                    'bsonType': 'object',
                    'required': ['username', 'email', 'password', 'full_name', 'mobile'],
                    'properties': {
                        'username': {
                            'bsonType': 'string',
                            'pattern': '^[a-zA-Z0-9_-]{3,16}$'  # Alphanumeric, underscore, hyphen, 3-16 chars
                        },
                        'email': {
                            'bsonType': 'string',
                            'pattern': '^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
                        },
                        'password': {'bsonType': 'string'},
                        'full_name': {'bsonType': 'string'},
                        'mobile': {
                            'bsonType': 'string',
                            'pattern': '^[0-9]{10}$'
                        },
                        'city': {'bsonType': 'string'},
                        'state': {'bsonType': 'string'},
                        'location': {
                            'bsonType': 'object',
                            'properties': {
                                'type': {'bsonType': 'string'},
                                'coordinates': {
                                    'bsonType': 'array',
                                    'items': {'bsonType': 'double'}
                                }
                            }
                        }
                    }
                }
            },
            'validationLevel': 'strict',
            'validationAction': 'error'
        })
        
    except Exception as e:
        logger.error(f"Error setting up database indices: {e}")
        # If the error is not about duplicate keys, re-raise it
        if not (isinstance(e, Exception) and 'E11000' in str(e)):
            raise

def is_valid_username(username):
    """Validate username format."""
    is_valid, error = validate_username(username)
    return is_valid

def is_valid_email(email):
    """Validate email format."""
    is_valid, error = validate_email(email)
    return is_valid

def is_valid_phone(phone):
    """Validate phone number format."""
    is_valid, error = validate_phone_number(phone)
    return is_valid

def check_username_availability(username):
    """Check if username is available."""
    return collection_enduser.find_one({'username': username}) is None

def check_email_availability(email):
    """Check if email is available."""
    # return collection_enduser.find_one({'email': email}) is None
    # Add debug logging
    try:
        existing_user = collection_enduser.find_one({'email': email})
        if existing_user:
            logger.info(f"Email {email} is already registered with user: {existing_user.get('username', 'unknown')}")
        else:
            logger.info(f"Email {email} is available for registration")
        
        # Count total users in collection for debugging
        total_users = collection_enduser.count_documents({})
        logger.info(f"Total users in database: {total_users}")
        
        return existing_user is None
    except Exception as e:
        logger.error(f"Error checking email availability: {e}")
        # Default to False (email not available) on error to prevent registration
        return False

def create_user(user_data):
    """Create a new user in the database."""
    try:
        # Validate username format
        if not is_valid_username(user_data.get('username', '')):
            raise ValueError("Invalid username format. Use 3-16 characters, letters, numbers, underscore, or hyphen.")
            
        # Check username availability
        if not check_username_availability(user_data.get('username')):
            raise ValueError("Username is already taken")
            
        # Validate email format
        if not is_valid_email(user_data.get('email', '')):
            raise ValueError("Invalid email format")
            
        # Validate mobile format
        if not is_valid_phone(user_data.get('mobile', '')):
            raise ValueError("Invalid mobile number format")
            
        # Hash the password
        user_data['password'] = generate_password_hash(user_data['password'])
        
        # Add timestamp
        user_data['created_at'] = datetime.datetime.now()
        user_data['last_login'] = datetime.datetime.now()
        
        # Ensure all required fields are present and not None
        required_fields = ['username', 'email', 'password', 'full_name', 'mobile']
        for field in required_fields:
            if not user_data.get(field):
                raise ValueError(f"Missing required field: {field}")
        
        # Convert location coordinates to float
        if user_data.get('location') and user_data['location'].get('coordinates'):
            user_data['location']['coordinates'] = [
                float(user_data['location']['coordinates'][0]),
                float(user_data['location']['coordinates'][1])
            ]
        
        # Insert user
        result = collection_enduser.insert_one(user_data)
        return user_data['username']  # Return username instead of ObjectId
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        raise

def authenticate_user(username_or_email, password):
    """
    Authenticate a user using either username or email.
    
    This function securely authenticates users by checking their credentials
    against the database, supporting both username and email-based login.
    
    Args:
        username_or_email (str): The username or email to authenticate
        password (str): The password to verify
        
    Returns:
        dict: User document if authentication succeeds, None otherwise
        
    Security features:
        - Uses secure password hashing with werkzeug
        - Supports both username and email authentication
        - Implements proper error handling and logging
        - Does not reveal whether username/email exists on failure
    """
    try:
        # Try to find user by username or email
        user = collection_enduser.find_one({
            '$or': [
                {'username': username_or_email},
                {'email': username_or_email}
            ]
        })
        
        if user and check_password_hash(user['password'], password):
            return user
        return None
    except Exception as e:
        logger.error(f"Error authenticating user: {e}")
        return None

## Weather API Configuration
# Weather API Configuration
WEATHER_API_KEY = "a88edcc7a885477aaa464656250104"  # Replace with actual API key
IP_API_URL = "http://ip-api.com/json/"
WEATHER_API_URL = "http://api.weatherapi.com/v1/current.json"

## Location and weather functions
def get_location_from_ip():
    """Gets location from IP-API as fallback only."""
    try:
        response = requests.get(IP_API_URL, timeout=10)
        response.raise_for_status()
        data = response.json()
        if not all(key in data for key in ['city', 'country', 'lat', 'lon']):
            raise ValueError("Missing required location data")
        return {
            'City': data.get('city', 'Unknown City'),
            'country': data.get('country', 'Unknown Country'),
            'state': data.get('regionName', 'Unknown State'),
            'area': data.get('district', ''),
            'lat': data.get('lat', 0.0),
            'lon': data.get('lon', 0.0),
            'ip': data.get('query', 'Unknown IP'),
            'zip': data.get('zip', '')
        }
    except (requests.exceptions.RequestException, ValueError) as e:
        logger.error(f"Error detecting location: {e}")
        return {
            'City': 'Unknown City', 
            'country': 'Unknown Country',
            'state': 'Unknown State',
            'area': '',
            'lat': 0.0,
            'lon': 0.0,
            'ip': 'Unknown IP',
            'zip': ''
        }

def get_weather(city):
    """
    Get current weather data for a specified city.
    
    This function fetches weather information from the WeatherAPI service,
    including temperature, humidity, wind, and air quality data.
    
    Args:
        city (str): The city name to get weather data for
        
    Returns:
        dict: Weather data dictionary if successful, None otherwise
        
    Features:
        - Includes air quality data when available
        - Handles API errors gracefully
        - Formats numeric values consistently
    """ 
    cache_duration = datetime.timedelta(minutes=30)
    rate_limit_key = f"weather_rate_limit_{city}"

    try:
        url = f"{WEATHER_API_URL}?key={WEATHER_API_KEY}&q={city}&aqi=yes"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        weather_data = response.json()
        current = weather_data.get("current", {})
        
        result = {
            "City": city,
            "Temperature": current.get("temp_c", ""),
            "Humidity": current.get("humidity", ""),
            "Wind_Speed": current.get("wind_kph", ""),
            "Wind_Direction": current.get("wind_dir", ""),
            "Condition": current.get("condition", {}).get("text", ""),
            "Pressure": current.get("pressure_mb", ""),
            "Visibility": current.get("vis_km", ""),
            "UV_Index": current.get("uv", ""),
            "Precipitation": current.get("precip_mm", ""),
            "Feels_Like": current.get("feelslike_c", ""),
            "Cloud_Cover": current.get("cloud", ""),
            "Gust_Speed": current.get("gust_kph", ""),
            "updated_at": datetime.datetime.now()
        }
        
        # Add air quality if available
        if "air_quality" in current:
            aqi = current.get("air_quality", {})
            result.update({
                "Air_Quality_Index": aqi.get("pm2_5", ""),
                "Air_Quality_Status": get_air_quality_status(aqi.get("pm2_5", 0))
            })
        
        return result
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching weather: {e}")
        return None

def get_air_quality_status(pm25):
    """Convert PM2.5 value to air quality status."""
    if pm25 <= 12:
        return "Good"
    elif pm25 <= 35.4:
        return "Moderate"
    elif pm25 <= 55.4:
        return "Unhealthy for Sensitive Groups"
    elif pm25 <= 150.4:
        return "Unhealthy"
    elif pm25 <= 250.4:
        return "Very Unhealthy"
    else:
        return "Hazardous"

def store_weather_data(weather_details):
    """Store weather data and return its ID for reference."""
    if weather_details:
        try:
            # Remove _id field if it exists
            if '_id' in weather_details:
                del weather_details['_id']
            
            # Ensure all numeric values are properly formatted
            for key, value in weather_details.items():
                if isinstance(value, (int, float)):
                    weather_details[key] = round(float(value), 2)
            
            result = collection_weather.update_one(
                {"City": weather_details["City"]}, 
                {"$set": weather_details}, 
                upsert=True
            )
            
            if result.upserted_id:
                return str(result.upserted_id)
            else:
                doc = collection_weather.find_one({"City": weather_details["City"]})
                return str(doc["_id"]) if doc else None
                
        except Exception as e:
            logger.error(f"Error storing weather data: {e}")
            return None
    return None


def validate_pattern(value, field_type):
    """
    Validates if a value matches any pattern defined for the given field type.
    
    Args:
        value (str): The value to validate
        field_type (str): The type of field to validate against (e.g., 'doctor_name', 'patient_name')
        
    Returns:
        bool: True if the value matches any pattern, False otherwise
    """
    try:
        # Get the appropriate collection based on field type
        collection = collection_store_validation
        
        # Find the mapping document for this field type
        mapping = collection.find_one({"standard_variable": field_type})
        if not mapping:
            logger.warning(f"No mapping found for field type: {field_type}")
            return True  # Default to True if no mapping found
        
        # Get patterns from the mapping
        patterns = mapping.get("patterns", [])
        if not patterns:
            return True  # Default to True if no patterns defined
        
        # Check if the value matches any pattern
        for pattern_dict in patterns:
            try:
                # Extract pattern from dictionary if it's a dict, otherwise use as is
                pattern_str = pattern_dict.get('regex') if isinstance(pattern_dict, dict) else pattern_dict
                if not isinstance(pattern_str, str):
                    continue
                    
                if re.search(pattern_str, value, re.IGNORECASE):
                    return True
            except re.error:
                continue
            except TypeError:
                continue
        
        # Check exclude patterns
        exclude_patterns = mapping.get("exclude_patterns", [])
        for pattern in exclude_patterns:
            if not isinstance(pattern, str):
                continue
                
            try:
                if re.search(pattern, value, re.IGNORECASE):
                    return False  # Value matches an exclude pattern
            except re.error:
                continue
            except TypeError:
                continue
        
        # If we have patterns but none matched, return False
        return False
    except Exception as e:
        logger.error(f"Error validating pattern for {field_type}: {e}")
        return True  # Default to True on error


def extract_store_details(content):
    """
    Extracts store details from the given markdown content using master validation mappings.
    
    This function parses unstructured text to extract structured information about
    patients, doctors, hospitals, and other entities using both database-defined
    patterns and fallback generic patterns.
    
    Args:
        content (str): Markdown content to parse
        
    Returns:
        dict: Dictionary of extracted fields with standardized values
        
    Process flow:
        1. Load field mappings from validation collection
        2. Try pattern matching using database-defined patterns
        3. Fall back to generic patterns for fields that weren't matched
        4. Clean and validate extracted values
        5. Standardize values using master data mappings
    """
    field_mappings = list(collection_store_validation.find())
    
    logger.info(f"Loaded {len(field_mappings)} field mappings from database")

    # Define standard fields
    standard_fields = {
        'patient_name': '',
        'relationship': '',  # This will be empty by default
        'patient_age': '',
        'doctor_name': '',
        'prescription_id': '',
        'phone_number': '',
        'email': '',
        'address': '',
        'hospital_name': '',
        'pharmacy_name': '',
        'lab_name': ''
    }
    
    # Generic patterns for common fields (fallback patterns)
    generic_patterns = {
        'phone_number': r'(?:Ph(?:one)?[.: ]*|(?:#|No)[.: ]*|Contact[.: ]*)?(\+?(?:\d[\d\- ]{8,}|\d{10}))',
        'email': r'[\w\.-]+@[\w\.-]+\.\w+',
        'doctor_name': r'(?:dr\.?|doctor|physician)[:\s]+([A-Za-z\s\.]+)(?=[\n,]|$)',
        'patient_name': r'(?:patient|name|pt\.?)[:\s]+([A-Za-z\s]+)(?=[\n,]|$)',
        'patient_age': r'(?:age|yrs?)[:\s]*(\d+)(?:\s*(?:years|yrs?|y)?)',
        'prescription_id': r'(?:prescription|rx|bill|invoice)\s*(?:no|number|id)[:\s]*([A-Za-z0-9-]+)',
        'address': r'(?:address|location|add)[:\s]+([^,\n]+(?:,[^,\n]+)*)',
        'hospital_name': r'(?:hospital|nursing\s+home|medical\s+center)[:\s]+([A-Za-z0-9\s]+)',
        'pharmacy_name': r'(?:pharmacy|chemist|medical|drug\s+store)[:\s]+([A-Za-z0-9\s]+)',
        'lab_name': r'(?:laboratory|lab|diagnostic)[:\s]+([A-Za-z0-9\s]+)'
    }
    
    lines = content.split('\n')
    logger.info("Processing content for store details")
    
    # First try MongoDB patterns
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        logger.debug(f"Processing line: {line}")
        
        # Try patterns from MongoDB mappings
        for mapping in field_mappings:
            standard_variable = mapping.get('standard_variable')
            if not standard_variable or standard_variable not in standard_fields:
                continue
                
            patterns = mapping.get('patterns', [])
            aliases = mapping.get('aliases', [])
            exclude_patterns = mapping.get('exclude_patterns', [])
            validation_rules = mapping.get('validation_rules', {})
            standard_values = mapping.get('standard_values', {})
            logger.info(f"Checking {len(patterns)} patterns for field {standard_variable}")
            # Skip if line matches any exclude pattern
            should_exclude = False
            for exclude_pattern in exclude_patterns:
                try:
                    if re.search(exclude_pattern, line, re.IGNORECASE):
                        should_exclude = True
                        break
                except (re.error, TypeError):
                    continue
            
            if should_exclude:
                continue

            # Try custom pattern matching
            for pattern_dict in patterns:
                try:
                    # Extract pattern from dictionary if it's a dict, otherwise use as is
                    pattern_str = pattern_dict.get('regex') if isinstance(pattern_dict, dict) else pattern_dict
                    if not isinstance(pattern_str, str):
                        logger.warning(f"Invalid pattern type for {standard_variable}: {type(pattern_str)}")
                        continue
                        
                    match = re.search(pattern_str, line, re.IGNORECASE)
                    if match:
                        value = match.group(1).strip() if match.groups() else match.group(0).strip()
                        logger.info(f"Matched {standard_variable} with database pattern: {value}")
                        # Additional validation for specific fields
                        if standard_variable == 'patient_age' and value:
                            try:
                                age = int(''.join(filter(str.isdigit, value)))
                                if 0 <= age <= 150:  # Reasonable age range
                                    standard_fields[standard_variable] = str(age)
                            except ValueError:
                                continue
                        elif standard_variable == 'phone_number' and value:
                            # Clean phone number and validate
                            clean_number = ''.join(filter(str.isdigit, value))
                            if len(clean_number) == 10:  # Standard phone number length
                                standard_fields[standard_variable] = clean_number
                        else:
                            if value and len(value) <= 100:  # Reasonable length check
                                standard_fields[standard_variable] = value
                        logger.debug(f"Custom pattern match for {standard_variable}: {value}")
                        break
                except re.error as e:
                    logger.warning(f"Invalid regex pattern '{pattern_str}': {e}")
                except TypeError as e:
                    logger.warning(f"Type error with pattern for {standard_variable}: {e}")
            
            # Try alias matching if no pattern match
            if not standard_fields[standard_variable]:
                for alias in aliases:
                    if not isinstance(alias, str):
                        continue
                    if alias.lower() in line.lower():
                        if ':' in line:
                            value = line.split(':')[1].strip()
                        else:
                            value = line[line.lower().find(alias.lower()) + len(alias):].strip()
                        if value and len(value) <= 100:  # Reasonable length check
                            standard_fields[standard_variable] = value
                            logger.debug(f"Alias match for {standard_variable}: {value}")
                            break
            
            # Standardize values if available
            if standard_fields[standard_variable] and standard_values:
                value = standard_fields[standard_variable]
                standardized = standard_values.get(value.lower(), value)
                if value != standardized:
                    standard_fields[standard_variable] = standardized
                    logger.debug(f"Standardized {standard_variable}: {value} -> {standardized}")
    
    # Then try generic patterns for any fields that are still empty
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        for field, pattern in generic_patterns.items():
            if not standard_fields[field]:  # Only try if field is still empty
                try:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        value = match.group(1).strip() if match.groups() else match.group(0).strip()
                        # Additional validation for specific fields
                        if field == 'patient_age' and value:
                            try:
                                age = int(''.join(filter(str.isdigit, value)))
                                if 0 <= age <= 150:  # Reasonable age range
                                    standard_fields[field] = str(age)
                            except ValueError:
                                continue
                        elif field == 'phone_number' and value:
                            # Clean phone number and validate
                            clean_number = ''.join(filter(str.isdigit, value))
                            if len(clean_number) == 10:  # Standard phone number length
                                standard_fields[field] = clean_number
                        else:
                            if value and len(value) <= 100:  # Reasonable length check
                                standard_fields[field] = value
                        logger.debug(f"Generic pattern match for {field}: {value}")
                except re.error as e:
                    logger.warning(f"Invalid generic pattern for {field}: {e}")
                except TypeError as e:
                    logger.warning(f"Type error with pattern for {field}: {e}")
    
    # Clean up and validate final values
    for field in standard_fields:
        value = standard_fields[field]
        if value:
            # Remove any unwanted characters or patterns
            value = re.sub(r'[^\w\s@.-]+', ' ', value)  # Keep only alphanumeric, @, ., - and spaces
            value = re.sub(r'\s+', ' ', value).strip()  # Normalize spaces
            if field == 'email' and '@' not in value:
                value = ''  # Clear invalid email
            standard_fields[field] = value
    
    logger.info(f"Final extracted store details: {standard_fields}")
    return standard_fields


## Matching Extracted Data (after saving edited changes on OCR extracted data) with Master Data
def match_medicine_batch(medicine_names):
    """Matches medicine details in batch from master_medicine collection."""
    if not medicine_names:
        return {}
    
    # Filter out empty names and normalize
    clean_names = [re.sub(r'[^\w\s]', '', name).strip().lower() 
                  for name in medicine_names if name.strip()]
    if not clean_names:
        return {}
    
    # Get medicine aliases from validation data
    medicine_aliases = get_field_aliases("medicine_name")
    
    # Create regex patterns for batch query with word boundaries and aliases
    medicine_regex_patterns = []
    for name in clean_names:
        # Start with the standard fields
        pattern = {
            "$or": [
                {"product_name": {'$regex': rf'\b{re.escape(name)}\b', '$options': 'i'}},
                {"generic_name": {'$regex': rf'\b{re.escape(name)}\b', '$options': 'i'}}
            ]
        }
        
        # Add alias fields to the pattern
        for alias in medicine_aliases:
            pattern["$or"].append({alias: {'$regex': rf'\b{re.escape(name)}\b', '$options': 'i'}})
            
        medicine_regex_patterns.append(pattern)
    
    # Fetch all matching medicines in one query
    # Don't exclude any fields - we want all details
    matched_medicines = list(collection_medicine.find({"$or": medicine_regex_patterns}))
    
    # Create lookup dictionary with both product and generic names
    medicine_lookup = {}
    for med in matched_medicines:
        product_name = med.get("product_name", "").lower()
        generic_name = med.get("generic_name", "").lower()
        medicine_lookup[product_name] = med
        if generic_name:
            medicine_lookup[generic_name] = med
            
        # Also add aliases to the lookup
        for alias in medicine_aliases:
            alias_value = med.get(alias, "").lower()
            if alias_value:
                medicine_lookup[alias_value] = med
    
    return medicine_lookup

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in kilometers using Haversine formula."""
    from math import radians, sin, cos, sqrt, atan2
    
    R = 6371  # Earth's radius in kilometers
    lat1, lon1, lat2, lon2 = map(radians, [float(lat1), float(lon1), float(lat2), float(lon2)])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c


def match_master_data(field_name, value, collection, location_data=None, pharmacy_location=None):
    """Matches a field value with location-based filtering and pattern validation."""
    if not value:
        return {}
        
    MAX_RADIUS_KM = {
        'urban': 5,    # Within city
        'suburban': 15, # Wider city area
        'rural': 30    # Rural areas
    }
    
    # Clean the value for matching
    value_clean = value.strip().lower()
    
    # Define field mappings for different collections
    field_mappings = {}
    if collection.name == 'master_doctor':
        field_mappings = {'doctor_name': ['doctor_name', 'name']}
    elif collection.name == 'master_pharmacy':
        field_mappings = {'pharmacy_name': ['Pharmacy Name', 'name', 'pharmacy_name', 'pharmacy name', 'Title', 'title']}  # Based on your MongoDB structure
    elif collection.name == 'master_hospital':
        field_mappings = {'hospital_name': ['Hospital Name', 'name', 'hospital_name', 'hospital name']}  # Based on your MongoDB structure
    elif collection.name == 'master_lab':
        field_mappings = {'lab_name': ['Lab Name', 'name', 'lab_name', 'lab name']}  # Based on your MongoDB structure
    
    # Get the actual field name to use based on collection
    actual_field_names = field_mappings.get(field_name, [field_name])
    
    # Special handling for doctors with location data
    if collection.name == 'master_doctor' and location_data:
        # First try exact match with name
        query = {"$or": [{field: {"$regex": value_clean, "$options": "i"}} for field in actual_field_names]}
        matching_doctors = list(collection.find(query))
        
        # If no exact matches, try fuzzy matching
        if not matching_doctors:
            all_doctors = list(collection.find())
            matching_doctors = []
            
            for doc in all_doctors:
                # Try all possible field names
                for field in actual_field_names:
                    doctor_name = doc.get(field, '').lower()
                    if doctor_name:
                        # Use Levenshtein distance for fuzzy matching
                        similarity = 100 - levenshtein_distance(value_clean, doctor_name)
                        similarity_ratio = similarity / max(len(value_clean), len(doctor_name)) * 100
                        
                        if similarity_ratio > 70:  # 70% similarity threshold
                            matching_doctors.append(doc)
                            break
        
        if not matching_doctors:
            print(f"No matching doctors found for '{value}'")
            return {}
            
        print(f"Found {len(matching_doctors)} matching doctors with name: {value}")
        for doc in matching_doctors:
            print(f"Doctor record: {doc}")
        
        # Get user location from location_data
        user_city = location_data.get('City', '').lower()
        user_lat = float(location_data.get('lat', 0))
        user_lon = float(location_data.get('lon', 0))
        
        # If pharmacy location is available, use it instead of user location
        if pharmacy_location:
            try:
                pharmacy_lat = float(pharmacy_location.get('latitude', 0))
                pharmacy_lon = float(pharmacy_location.get('longitude', 0))
                if pharmacy_lat != 0 and pharmacy_lon != 0:
                    user_lat = pharmacy_lat
                    user_lon = pharmacy_lon
                    print(f"Using pharmacy location: ({user_lat}, {user_lon})")
            except (ValueError, TypeError):
                pass
        
        print(f"User location - City: {user_city}, Coordinates: ({user_lat}, {user_lon})")
        
        # First try to match by city
        doctors_in_city = [d for d in matching_doctors 
                          if d.get('city', '').lower().strip() == user_city]
        
        if doctors_in_city:
            print(f"Found {len(doctors_in_city)} doctors with name '{value}' in {user_city}")
            
            # Initialize doctors_with_exp list outside the loop
            doctors_with_exp = []
            
            for doc in doctors_in_city:
                exp_str = doc.get('experience', '0')
                # Extract just the number from strings like "15yrs"
                exp_value = int(re.search(r'(\d+)', exp_str).group(1)) if re.search(r'(\d+)', exp_str) else 0
                doctors_with_exp.append((exp_value, doc))
            
            if doctors_with_exp:
                # Sort by experience (highest first)
                doctors_with_exp.sort(key=lambda x: x[0], reverse=True)
                most_experienced = doctors_with_exp[0]
                # Use the first field name that exists in the document
                for field in actual_field_names:
                    if field in most_experienced[1]:
                        print(f"Selected most experienced doctor: {most_experienced[1].get(field)} ({most_experienced[0]} years)")
                        break
                return most_experienced[1]
            
            # If all else fails, return the first doctor in the city
            print(f"Selected first matching doctor in {user_city}")
            return doctors_in_city[0]
        
        # If no city match, try state match
        doctors_in_state = [d for d in matching_doctors 
                           if d.get('state', '').lower().strip() == location_data.get('state', '').lower().strip()]
        if doctors_in_state:
            print(f"Selected doctor from same state")
            return doctors_in_state[0]
            
        # If no state match, return first matching doctor
        print(f"Selected first matching doctor")
        return matching_doctors[0]
    
    # For non-doctor collections or no location data
    print(f"Searching for {field_name} '{value}' in {collection.name}")
    
    # Try all possible field names for this collection
    for actual_field in actual_field_names:
        print(f"Trying field '{actual_field}' in {collection.name}")
        
        # Try direct exact match first (case insensitive)
        exact_match = collection.find_one({actual_field: {'$regex': f'^{re.escape(value_clean)}$', '$options': 'i'}})
        if exact_match:
            print(f"Found exact match for {actual_field}: {exact_match.get(actual_field)}")
            return exact_match
        
        # Try contains match (case insensitive)
        contains_match = collection.find_one({actual_field: {'$regex': re.escape(value_clean), '$options': 'i'}})
        if contains_match:
            print(f"Found contains match for {actual_field}: {contains_match.get(actual_field)}")
            return contains_match
    
    # If no direct matches, try manual search through all records
    all_records = list(collection.find())
    best_match = None
    best_score = 0
    
    for record in all_records:
        # Try each possible field name
        for actual_field in actual_field_names:
            record_value = record.get(actual_field, '')
            if record_value and record_value.lower() == value_clean:
                print(f"Found manual exact match on field {actual_field}: {record_value}")
                return record
            
            # Try fuzzy matching
            if record_value:
                record_value_clean = record_value.lower()
                
                # Calculate similarity
                similarity_ratio = fuzz.ratio(value_clean, record_value_clean)
                if similarity_ratio > best_score:
                    best_score = similarity_ratio
                    best_match = record
                    print(f"New best match on field {actual_field}: {record_value} with score {similarity_ratio}")
    
    # Return the best match if it's above our threshold
    if best_score > 80:  # 80% similarity threshold
        print(f"Found fuzzy match for '{value}' with {best_score}% similarity")
        return best_match
    
    print(f"Warning: {field_name.replace('_', ' ').title()} '{value}' not found in master data")
    return {}

def debug_entity_search(entity_type, entity_name):
    """Debug function to search for entities directly in the database."""
    if entity_type == "pharmacy":
        collection = collection_pharmacy
        possible_field_names = ["pharmacy_name", "name", "Pharmacy Name", 'pharmacy name', 'Title', 'title']
    elif entity_type == "hospital":
        collection = collection_hospital
        possible_field_names = ["hospital_name", "name", "Hospital Name", 'hospital name', 'Title', 'title']
    elif entity_type == "lab":
        collection = collection_lab
        possible_field_names = ["lab_name", "name", "Lab Name", 'lab name', 'Title', 'title']
    elif entity_type == "doctor":
        collection = collection_doctor
        possible_field_names = ["doctor_name", "name", "Doctor Name", 'doctor name', 'Title', 'title']
    else:
        print(f"Unknown entity type: {entity_type}")
        return None
    
    # Print all records in the collection
    all_records = list(collection.find())
    print(f"All {entity_type} records ({len(all_records)}):")
    
    # Find which field name exists in the first record
    field_name = None
    if all_records:
        first_record = all_records[0]
        for possible_field in possible_field_names:
            if possible_field in first_record:
                field_name = possible_field
                print(f"Using field name '{field_name}' for {entity_type}")
                break
    
    if not field_name and all_records:
        print(f"Available fields in {entity_type} records:")
        for key in all_records[0].keys():
            print(f"  - {key}")
        field_name = "Lab Name"  # Default based on your MongoDB structure
    
    # Print all records with their name field
    for record in all_records:
        record_name = record.get(field_name, 'No name')
        print(f"  - {record_name} (ID: {record.get('_id')})")
    
    # Try exact match
    for possible_field in possible_field_names:
        exact_match = collection.find_one({possible_field: entity_name})
        if exact_match:
            print(f"Found exact match for {entity_name} on field {possible_field}: {exact_match}")
            return exact_match
    
    # Try case-insensitive match
    for possible_field in possible_field_names:
        case_insensitive = collection.find_one({possible_field: {'$regex': f'^{re.escape(entity_name)}$', '$options': 'i'}})
        if case_insensitive:
            print(f"Found case-insensitive match for {entity_name} on field {possible_field}: {case_insensitive}")
            return case_insensitive
    
    # Try contains match
    for possible_field in possible_field_names:
        contains_match = collection.find_one({possible_field: {'$regex': re.escape(entity_name), '$options': 'i'}})
        if contains_match:
            print(f"Found contains match for {entity_name} on field {possible_field}: {contains_match}")
            return contains_match
    
    print(f"No match found for {entity_type} '{entity_name}'")
    return None

# Add this helper function at the top of the file
def levenshtein_distance(s1, s2):
    """Calculate the Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def fetch_master_data_batch(data_dict, collections_map, location_data=None):
    """Fetch master data in batch for multiple collections."""
    results = {}
    
    # Debug: Print all entity names we're looking for
    print(f"Searching for entities: {data_dict}")
    
    # First get pharmacy location if available
    pharmacy_location = None
    if "pharmacy_name" in data_dict and data_dict["pharmacy_name"]:
        pharmacy_name = data_dict["pharmacy_name"]
        print(f"Looking up pharmacy: '{pharmacy_name}'")
        
        # Debug: Search directly in the database
        debug_entity_search("pharmacy", pharmacy_name)
        
        pharmacy_result = match_master_data(
            collections_map["pharmacy_name"]["field"],
            pharmacy_name,
            collections_map["pharmacy_name"]["collection"]
        )
        if pharmacy_result:
            # Check for latitude/longitude using different possible field names
            lat_fields = ['latitude', 'Latitude', 'lat']
            lon_fields = ['longitude', 'Longitude', 'lon', 'lng']
            
            has_coords = False
            for lat_field in lat_fields:
                for lon_field in lon_fields:
                    if lat_field in pharmacy_result and lon_field in pharmacy_result:
                        pharmacy_location = pharmacy_result
                        results["pharmacy_name"] = pharmacy_result
                        print(f"Found pharmacy with coordinates using fields {lat_field}/{lon_field}")
                        has_coords = True
                        break
                if has_coords:
                    break
            
            if not has_coords:
                print(f"Found pharmacy but missing coordinates")
                results["pharmacy_name"] = pharmacy_result
        else:
            print(f"Warning: Pharmacy '{pharmacy_name}' not found in master data")
    
    # Then process doctor with both user location and pharmacy location
    if "doctor_name" in data_dict and data_dict["doctor_name"]:
        doctor_name = data_dict["doctor_name"]
        print(f"Looking up doctor: '{doctor_name}'")
        doctor_result = match_master_data(
            collections_map["doctor_name"]["field"],
            doctor_name,
            collections_map["doctor_name"]["collection"],
            location_data=location_data,
            pharmacy_location=pharmacy_location  # Will be None if pharmacy not found or missing coordinates
        )
        if doctor_result:
            results["doctor_name"] = doctor_result
            # Use the first field name that exists in the document
            for field in ['doctor_name', 'name', 'Doctor Name', 'doctor name', 'Title', 'title']:
                if field in doctor_result:
                    print(f"Found doctor: {doctor_result.get(field)}")
                    break
    
    # Process hospital
    if "hospital_name" in data_dict and data_dict["hospital_name"]:
        hospital_name = data_dict["hospital_name"]
        print(f"Looking up hospital: '{hospital_name}'")
        
        # Debug: Search directly in the database
        debug_entity_search("hospital", hospital_name)
        
        hospital_result = match_master_data(
            collections_map["hospital_name"]["field"],
            hospital_name,
            collections_map["hospital_name"]["collection"]
        )
        if hospital_result:
            results["hospital_name"] = hospital_result
            # Use the first field name that exists in the document
            for field in ['hospital_name', 'name', 'Hospital Name', 'hospital name', 'Title', 'title']:
                if field in hospital_result:
                    print(f"Found hospital: {hospital_result.get(field)}")
                    break
    
    # Process lab
    if "lab_name" in data_dict and data_dict["lab_name"]:
        lab_name = data_dict["lab_name"]
        print(f"Looking up lab: '{lab_name}'")
        
        # Debug: Search directly in the database
        debug_entity_search("lab", lab_name)
        
        lab_result = match_master_data(
            collections_map["lab_name"]["field"],
            lab_name,
            collections_map["lab_name"]["collection"]
        )
        if lab_result:
            results["lab_name"] = lab_result
            # Use the first field name that exists in the document
            for field in ['lab_name', 'name', 'Lab Name', 'lab name', 'Title', 'title']:
                if field in lab_result:
                    print(f"Found lab: {lab_result.get(field)}")
                    break
    
    return results

def store_purchase_data(df, store_details, user_data=None):
    """
    Store purchase data safely with proper error handling.
    
    Args:
        df (DataFrame): DataFrame containing purchase data
        store_details (dict): Dictionary with store and patient details
        user_data (dict, optional): User information if available
        
    Returns:
        tuple: (success_flag, message)
    """
    try:
        if df is None or df.empty:
            logger.warning("No purchase data to save - DataFrame is empty")
            return False, "No purchase data to save"
            
        # Convert DataFrame to list of dictionaries for MongoDB
        purchase_records = df.to_dict('records')
        
        # Add store details to each record
        for record in purchase_records:
            # Add store details
            for key, value in store_details.items():
                if value:  # Only add non-empty values
                    record[f"store_{key}"] = value
                    
            # Add user details if available
            if user_data:
                record["user_id"] = user_data.get("username", "")
                record["user_email"] = user_data.get("email", "")
                
            # Add timestamp if not present
            if "timestamp" not in record:
                record["timestamp"] = datetime.datetime.now()
                
        # Insert records into purchase collection
        if purchase_records:
            result = collection_purchase.insert_many(purchase_records)
            logger.info(f"Saved {len(result.inserted_ids)} purchase records")
            return True, f"Saved {len(result.inserted_ids)} purchase records"
        else:
            return False, "No records to save"
            
    except Exception as e:
        logger.error(f"Error saving purchase data: {str(e)}")
        return False, f"Error saving purchase data: {str(e)}"

## Parsing the markdown table
def parse_markdown_table(content):
    """
    Parses a markdown table using master validation mappings.
    
    This function extracts structured tabular data from markdown text,
    mapping column headers to standardized field names using database-defined
    validation rules.
    
    Args:
        content (str): Markdown content containing tables
        
    Returns:
        DataFrame: Pandas DataFrame with standardized column names and validated data
        
    Process flow:
        1. Extract table content from markdown
        2. Identify headers and map to standard column names
        3. Parse rows into structured data
        4. Apply data type conversions and validations
        5. Return standardized DataFrame
        
    Raises:
        ValueError: If column validation mappings are not found in database
    """
    logger.info("Starting markdown table parsing")
    
    try:
        # Get column mappings from database
        column_mappings = list(collection_table_validation.find())
        if not column_mappings:
            logger.error("No column mappings found in database")
            raise ValueError("Column validation mappings not found in database")
        
        # Define standard columns we want to keep
        standard_columns = [
            'sr_no', 'medicine_name', 'dosage', 'batch_number', 'manufacturer_name',
            'manufactured_date', 'expiry_date', 'quantity', 'mrp', 'rate', 'amount'
        ]
        
        # Find table content in markdown
        table_matches = re.findall(r'\|(.*?)\|\n', content, re.MULTILINE)
        if not table_matches:
            logger.warning("No table content found in markdown")
            return pd.DataFrame(columns=standard_columns)
        
        # Clean up table matches and remove separator lines
        table_matches = [line.strip() for line in table_matches if line.strip()]
        table_matches = [line for line in table_matches if not re.match(r'^[-:|]+$', line.strip())]
        
        if not table_matches:
            logger.warning("No valid table rows found after cleaning")
            return pd.DataFrame(columns=standard_columns)
        
        # Get headers from first row
        headers = [col.strip().lower() for col in table_matches[0].split('|') if col.strip()]
        logger.info(f"Original headers found: {headers}")
        
        # Create header mapping using database mappings
        header_mapping = {}
        for mapping in column_mappings:
            standard_column = mapping.get('standard_column')
            if not standard_column or standard_column not in standard_columns:
                continue
                
            aliases = mapping.get('aliases', [])
            patterns = mapping.get('patterns', [])
            
            for idx, header in enumerate(headers):
                # Clean header for comparison
                clean_header = re.sub(r'[^a-zA-Z0-9\s]', '', header).lower().strip()
                
                # Check if header matches any alias
                if any(str(alias).lower().strip() == clean_header for alias in aliases):
                    header_mapping[idx] = standard_column
                    logger.info(f"Mapped column {header} to {standard_column} via alias")
                    break
                
                # Check if header matches any pattern
                for pattern in patterns:
                    try:
                        pattern_str = pattern.get('regex') if isinstance(pattern, dict) else pattern
                        if re.search(pattern_str, clean_header, re.IGNORECASE):
                            header_mapping[idx] = standard_column
                            logger.info(f"Mapped column {header} to {standard_column} via pattern")
                            break
                    except (re.error, AttributeError) as e:
                        logger.warning(f"Invalid pattern in database: {e}")
        
        logger.info(f"Header mapping created: {header_mapping}")
        
        # Process data rows
        data_rows = []
        expected_col_count = len(headers)
        for row in table_matches[1:]:  # Skip header row
            cells = [cell.strip() for cell in row.split('|')]
            # Pad or truncate cells to match header count
            if len(cells) < expected_col_count:
                cells += [''] * (expected_col_count - len(cells))
            elif len(cells) > expected_col_count:
                cells = cells[:expected_col_count]

            # Remove empty leading/trailing cells (from split)
            while cells and cells[0] == '':
                cells.pop(0)
            while cells and cells[-1] == '':
                cells.pop()

            # Pad again if needed after pop
            if len(cells) < expected_col_count:
                cells += [''] * (expected_col_count - len(cells))

            if not cells:
                continue

            # Create a new row with standard columns
            new_row = {col: '' for col in standard_columns}
            
            # Map cells to correct columns
            for idx, cell in enumerate(cells):
                if idx in header_mapping:
                    std_col = header_mapping[idx]
                    new_row[std_col] = cell
            
            # Apply validation rules from database
            for mapping in column_mappings:
                std_col = mapping.get('standard_column')
                if not std_col or std_col not in new_row:
                    continue
                
                value = new_row[std_col]
                if not value:
                    continue
                
                # Apply validation patterns
                validation_patterns = mapping.get('validation_patterns', [])
                for pattern in validation_patterns:
                    try:
                        pattern_str = pattern.get('regex') if isinstance(pattern, dict) else pattern
                        if pattern_str:
                            match = re.match(pattern_str, value, re.IGNORECASE)
                            if match and match.groups():
                                new_row[std_col] = match.group(1)  # Use first captured group
                    except (re.error, AttributeError) as e:
                        logger.warning(f"Invalid validation pattern in database: {e}")
                
                # Apply standardization rules
                standard_values = mapping.get('standard_values', {})
                if standard_values and isinstance(standard_values, dict):
                    value_lower = value.lower()
                    if value_lower in standard_values:
                        new_row[std_col] = standard_values[value_lower]
            
            # Special handling for manufacturer name and date
            if 'manufacturer_name' in new_row and 'manufactured_date' in new_row:
                mfg_value = new_row['manufactured_date']
                if mfg_value and not re.match(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}', mfg_value):
                    new_row['manufacturer_name'] = mfg_value
                    new_row['manufactured_date'] = ''
            
            data_rows.append(new_row)
        
        # Create DataFrame
        df = pd.DataFrame(data_rows, columns=standard_columns)
        
        # Clean up the data
        df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        df = df.fillna('')
        
        # Add serial numbers if missing
        if df['sr_no'].isnull().all() or df['sr_no'].eq('').all():
            df['sr_no'] = range(1, len(df) + 1)
        
        logger.info(f"Created DataFrame with {len(df)} rows")
        return df
        
    except Exception as e:
        logger.error(f"Error parsing markdown table: {e}")
        return pd.DataFrame(columns=standard_columns)

def merge_medical_store_data(df, weather_id):
    """Merges extracted table and adds weather ID reference, handling unnamed columns by position."""

    # Define standard columns (expected order before adding metadata)
    standard_columns = [
        'sr_no', 'medicine_name', 'dosage', 'batch_number', 'manufacturer_name',
        'manufactured_date', 'expiry_date', 'quantity', 'mrp', 'rate', 'amount'
    ]

    # Auto-map unnamed columns to their correct standard position
    unnamed_columns = [col for col in df.columns if col.startswith('unnamed_col_')]
    if unnamed_columns:
        for col in unnamed_columns:
            col_position = list(df.columns).index(col)
            if col_position < len(standard_columns):
                correct_name = standard_columns[col_position]
                # Only rename if the correct name doesn't already exist
                if correct_name not in df.columns:
                    df.rename(columns={col: correct_name}, inplace=True)

    # Add weather reference and timestamp fields
    df['weather_id'] = weather_id
    current_time = datetime.datetime.now()
    df['timestamp'] = current_time
    df['created_date'] = current_time.strftime('%Y-%m-%d')
    df['created_time'] = current_time.strftime('%H:%M:%S')

    # Dynamically build final column order
    column_order = standard_columns + ['weather_id', 'timestamp', 'created_date', 'created_time']

    # Ensure all required columns are present
    for col in column_order:
        if col not in df.columns:
            df[col] = ''

    return df[column_order]


def enrich_ocr_data(df):
    """Enrich OCR data with matched master records using batch operations."""
    
    # Get location data for doctor matching
    location_data = get_location_from_ip()

    # Extract key data for batch lookups
    # The issue is here - we need to check both column formats
    doctor_name = ""
    pharmacy_name = ""
    hospital_name = ""
    lab_name = ""

    # Extract key data for batch lookups
    # Try both formats of column names (with and without spaces)
    if 'Doctor Name' in df.columns and not df.empty:
        doctor_name = df['Doctor Name'].iloc[0]
    elif 'doctor_name' in df.columns and not df.empty:
        doctor_name = df['doctor_name'].iloc[0]
        
    if 'Pharmacy Name' in df.columns and not df.empty:
        pharmacy_name = df['Pharmacy Name'].iloc[0]
    elif 'pharmacy_name' in df.columns and not df.empty:
        pharmacy_name = df['pharmacy_name'].iloc[0]
        
    if 'Hospital Name' in df.columns and not df.empty:
        hospital_name = df['Hospital Name'].iloc[0]
    elif 'hospital_name' in df.columns and not df.empty:
        hospital_name = df['hospital_name'].iloc[0]
        
    if 'Lab Name' in df.columns and not df.empty:
        lab_name = df['Lab Name'].iloc[0]
    elif 'lab_name' in df.columns and not df.empty:
        lab_name = df['lab_name'].iloc[0]
    
    logger.info(f"enrich_ocr_data: doctor_name={doctor_name}, pharmacy_name={pharmacy_name}, hospital_name={hospital_name}, lab_name={lab_name}")
    logger.info(f"enrich_ocr_data: DataFrame columns: {df.columns.tolist()}")

    # Batch fetch master data for entity references
    master_data = fetch_master_data_batch(
        {
            "doctor_name": doctor_name,
            "pharmacy_name": pharmacy_name,
            "hospital_name": hospital_name,
            "lab_name": lab_name
        },
        {
            "doctor_name": {"field": "doctor_name", "collection": collection_doctor},
            "pharmacy_name": {"field": "pharmacy_name", "collection": collection_pharmacy},
            "hospital_name": {"field": "hospital_name", "collection": collection_hospital},
            "lab_name": {"field": "lab_name", "collection": collection_lab}
        },
        location_data=location_data
    )
    logger.info(f"enrich_ocr_data: master_data fetched: {master_data}")

    # Add entity IDs to the DataFrame
    if "doctor_name" in master_data and master_data["doctor_name"]:
        df["doctor_id"] = str(master_data["doctor_name"].get("_id", ""))
    
    if "pharmacy_name" in master_data and master_data["pharmacy_name"]:
        df["pharmacy_id"] = str(master_data["pharmacy_name"].get("_id", ""))
    
    if "hospital_name" in master_data and master_data["hospital_name"]:
        df["hospital_id"] = str(master_data["hospital_name"].get("_id", ""))
    
    if "lab_name" in master_data and master_data["lab_name"]:
        df["lab_id"] = str(master_data["lab_name"].get("_id", ""))

    # Add all details for doctor, pharmacy, hospital, lab (not just IDs)
    entity_keys = [
        ("doctor_name", "master_doctor_"),
        ("pharmacy_name", "master_pharmacy_"),
        ("hospital_name", "master_hospital_"),
        ("lab_name", "master_lab_"),
    ]
    for entity_key, prefix in entity_keys:
        entity_data = master_data.get(entity_key)
        logger.info(f"enrich_ocr_data: entity_key={entity_key}, entity_data={entity_data}")
        if entity_data:
            for key, value in entity_data.items():
                if key != "_id":  # Skip the ID as we already added it
                    col_name = f"{prefix}{key}"
                    df[col_name] = value   
    
    # Get unique medicine names for batch lookup
    medicine_names = []
    if 'medicine_name' in df.columns:
        medicine_names = df["medicine_name"].dropna().unique().tolist()
    medicine_lookup = match_medicine_batch(medicine_names)
    
    # Apply medicine matches to dataframe rows and add all medicine details
    for index, row in df.iterrows():
        medicine_name = row.get("medicine_name", "").strip().lower()
        if medicine_name:
            # Try exact match first
            matched_data = medicine_lookup.get(medicine_name)
            
            # If not found, try to find the closest match
            if not matched_data:
                for key in medicine_lookup:
                    if medicine_name in key or key in medicine_name:
                        matched_data = medicine_lookup[key]
                        break
            
            # Add all medicine details instead of just the ID
            if matched_data:
                # Store the ID for reference
                df.at[index, "medicine_id"] = matched_data.get("_id", "")
                
                # Add all medicine details as new columns
                for key, value in matched_data.items():
                    if key != "_id":  # Skip the ID as we already added it
                        column_name = f"master_{key}"  # Prefix to avoid column name conflicts
                        df.at[index, column_name] = value
    
    return df

def store_ocr_data_with_relationships(user_data, store_details, enriched_data):
    """
    Store OCR data in multiple collections based on relationship.
    
    Args:
        user_data: Dictionary containing user information (username, email)
        store_details: Dictionary containing store and patient details
        enriched_data: DataFrame containing the processed OCR data
    """
    try:
        # Convert enriched_data to list of dictionaries for MongoDB
        bills_data = enriched_data.to_dict('records')
        
        # Get latitude and longitude from store_details
        latitude = float(store_details.get('latitude', 0))
        longitude = float(store_details.get('longitude', 0))
        
        # Create bill document with current timestamp and location
        bill_document = {
            "upload_time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "store": store_details.get('pharmacy_name', ''),
            "medicine_list": [bill.get('medicine_name', '') for bill in bills_data],
            "amount": sum(float(bill.get('amount', 0)) for bill in bills_data),
            "upload_location": {
                "lat": latitude,
                "lng": longitude
            },
            "original_data": bills_data  # Store complete bill details
        }

        relationship = store_details.get('relationship', '').lower()
        
        if relationship == 'self':
            # Store in master_enduser collection
            update_result = collection_enduser.update_one(
                {"username": user_data['username']},
                {
                    "$push": {"bills": bill_document},
                    "$set": {
                        "last_bill_upload": datetime.datetime.now(),
                        "last_activity": datetime.datetime.now(),
                        "location": {
                            "lat": latitude,
                            "lng": longitude
                        }
                    }
                }
            )
            if not update_result.modified_count:
                logger.error(f"Failed to update master_enduser for user {user_data['username']}")
                raise Exception("Failed to update user data")
                
        else:
            # Check if user has reached maximum patient limit
            existing_patient_doc = collection_patient.find_one({"_id": user_data['username']})
            patient_count = len(existing_patient_doc.get('patients', [])) if existing_patient_doc else 0
            
            if patient_count >= 10:
                raise Exception("Maximum patient limit (10) reached")
            
            # Check if patient already exists
            existing_patient = collection_patient.find_one(
                {
                    "_id": user_data['username'],
                    "patients": {
                        "$elemMatch": {
                            "name": store_details['patient_name'],
                            "relationship": relationship
                        }
                    }
                }
            )
            
            if existing_patient:
                # Add bill to existing patient and update location
                update_result = collection_patient.update_one(
                    {
                        "_id": user_data['username'],
                        "patients": {
                            "$elemMatch": {
                                "name": store_details['patient_name'],
                                "relationship": relationship
                            }
                        }
                    },
                    {
                        "$push": {"patients.$.bills": bill_document},
                        "$set": {
                            "patients.$.last_bill_upload": datetime.datetime.now(),
                            "patients.$.location": {
                                "lat": latitude,
                                "lng": longitude
                            }
                        }
                    }
                )
            else:
                # Create new patient entry with location
                new_patient = {
                    "patient_id": str(uuid.uuid4())[:8],
                    "name": store_details['patient_name'],
                    "relationship": relationship,
                    "location": {
                        "lat": latitude,
                        "lng": longitude
                    },
                    "bills": [bill_document],
                    "created_at": datetime.datetime.now(),
                    "last_bill_upload": datetime.datetime.now()
                }
                
                update_result = collection_patient.update_one(
                    {"_id": user_data['username']},
                    {
                        "$push": {"patients": new_patient}
                    },
                    upsert=True
                )
            
            if not update_result.modified_count and not update_result.upserted_id:
                logger.error(f"Failed to update master_patient for user {user_data['username']}")
                raise Exception("Failed to update patient data")
            
            # Update patient count in master_enduser
            updated_patient_doc = collection_patient.find_one({"_id": user_data['username']})
            current_patient_count = len(updated_patient_doc.get('patients', [])) if updated_patient_doc else 0
            
            collection_enduser.update_one(
                {"username": user_data['username']},
                {
                    "$set": {
                        "patient_count": current_patient_count,
                        "last_activity": datetime.datetime.now()
                    }
                }
            )
        
        # Store in ocr_data collection as before
        mongo_data = enriched_data.to_dict('records')
        result = collection_ocr.insert_many(mongo_data)
        
        return True, "Data stored successfully"
        
    except Exception as e:
        logger.error(f"Error storing OCR data: {e}")
        return False, str(e)

def get_user_patients(username):
    """
    Get list of patients for a user.
    """
    try:
        patient_data = collection_patient.find_one({"_id": username})
        if patient_data:
            return patient_data.get('patients', [])
        return []
    except Exception as e:
        logger.error(f"Error fetching patient data: {e}")
    return []

def get_user_bills(username):
    """
    Get all bills for a user (both personal and patients).
    Returns bills sorted by upload time with properly formatted data.
    """
    try:
        # Get user's personal bills
        user_data = collection_enduser.find_one({"username": username})
        personal_bills = []
        if user_data and 'bills' in user_data:
            for bill in user_data['bills']:
                try:
                    # Format personal bill data
                    formatted_bill = {
                        'upload_time': bill.get('upload_time'),
                        'store': bill.get('store', 'Unknown Store'),
                        'patient_name': user_data.get('full_name', 'Self'),  # Use user's name for personal bills
                        'relationship': 'self',
                        'amount': bill.get('amount', 0),
                        'medicine_list': bill.get('medicine_list', []),
                        'original_data': bill.get('original_data', [])
                    }
                    
                    # Convert ObjectId to string in original_data if present
                    if formatted_bill['original_data']:
                        for item in formatted_bill['original_data']:
                            if '_id' in item:
                                item['_id'] = str(item['_id'])
                            if 'medicine_id' in item:
                                item['medicine_id'] = str(item['medicine_id'])
                    
                    # Add location if available
                    if 'upload_location' in bill:
                        formatted_bill['location'] = bill['upload_location']
                    
                    personal_bills.append(formatted_bill)
                except Exception as e:
                    logger.error(f"Error formatting personal bill: {e}")
                    continue
        
        # Get patients' bills
        patient_data = collection_patient.find_one({"_id": username})
        patient_bills = []
        if patient_data and 'patients' in patient_data:
            for patient in patient_data['patients']:
                for bill in patient.get('bills', []):
                    try:
                        # Format patient bill data
                        formatted_bill = {
                            'upload_time': bill.get('upload_time'),
                            'store': bill.get('store', 'Unknown Store'),
                            'patient_name': patient.get('name', 'Unknown Patient'),
                            'relationship': patient.get('relationship', 'other'),
                            'amount': bill.get('amount', 0),
                            'medicine_list': bill.get('medicine_list', []),
                            'original_data': bill.get('original_data', []),
                            'patient_id': patient.get('patient_id')
                        }
                        
                        # Convert ObjectId to string in original_data if present
                        if formatted_bill['original_data']:
                            for item in formatted_bill['original_data']:
                                if '_id' in item:
                                    item['_id'] = str(item['_id'])
                                if 'medicine_id' in item:
                                    item['medicine_id'] = str(item['medicine_id'])
                        
                        # Add location if available
                        if 'upload_location' in bill:
                            formatted_bill['location'] = bill['upload_location']
                        
                        patient_bills.append(formatted_bill)
                    except Exception as e:
                        logger.error(f"Error formatting patient bill: {e}")
                        continue
        
        # Combine all bills
        all_bills = personal_bills + patient_bills
        
        # Parse upload_time strings to datetime objects for sorting
        for bill in all_bills:
            try:
                if isinstance(bill['upload_time'], str):
                    bill['upload_time'] = datetime.datetime.fromisoformat(bill['upload_time'].replace('Z', '+00:00'))
            except (ValueError, TypeError, KeyError) as e:
                logger.error(f"Error parsing upload time: {e}")
                bill['upload_time'] = datetime.datetime.now()  # Use current time as fallback
        
        # Sort by upload time (most recent first)
        all_bills.sort(key=lambda x: x['upload_time'], reverse=True)
        
        # Format dates back to ISO format strings for JSON serialization
        for bill in all_bills:
            if isinstance(bill['upload_time'], datetime.datetime):
                bill['upload_time'] = bill['upload_time'].isoformat()
        
        return all_bills
    except Exception as e:
        logger.error(f"Error fetching bills: {e}")
        return []

def process_document(input_doc_path):
    """Process document with OCR and enrich with master data."""
    start_time = time.time()
    
    try:
        # Get absolute paths for directories
        base_dir = os.path.dirname(os.path.abspath(__file__))
        upload_dir = os.path.join(base_dir, 'uploads')
        markdown_dir = os.path.join(upload_dir, 'markdown')
        
        # Create directories if they don't exist
        os.makedirs(upload_dir, exist_ok=True)
        os.makedirs(markdown_dir, exist_ok=True)
        
        logger.info(f"Base directory: {base_dir}")
        logger.info(f"Upload directory: {upload_dir}")
        logger.info(f"Markdown directory: {markdown_dir}")
        
        # Set up document converter with OCR capability
        accelerator_options = AcceleratorOptions(num_threads=16, device=AcceleratorDevice.CUDA)
        pipeline_options = PdfPipelineOptions()
        pipeline_options.accelerator_options = accelerator_options
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True

        converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF, InputFormat.IMAGE, InputFormat.CSV],
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
                InputFormat.DOCX: WordFormatOption(pipeline_cls=SimplePipeline)
            }
        )

        settings.debug.profile_pipeline_timings = True

        # Convert document and extract content
        logger.info(f"Converting document: {input_doc_path}")
        conversion_result = converter.convert(input_doc_path)
        extracted_content = conversion_result.document.export_to_markdown()
        logger.info(f"Extracted content length: {len(extracted_content)}")
        
        # Save markdown file
        try:
            input_filename = os.path.splitext(os.path.basename(input_doc_path))[0]
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            markdown_filename = f"{input_filename}_{timestamp}.md"
            markdown_path = os.path.join(markdown_dir, markdown_filename)
            
            logger.info(f"Attempting to save markdown to: {markdown_path}")
            with open(markdown_path, "w", encoding="utf-8") as md_file:
                md_file.write(extracted_content)
                md_file.flush()
            
            if os.path.exists(markdown_path):
                file_size = os.path.getsize(markdown_path)
                logger.info(f"Successfully saved markdown file. Size: {file_size} bytes")
            else:
                logger.error(f"Failed to save markdown file at: {markdown_path}")
                
        except Exception as e:
            logger.error(f"Error while saving markdown file: {str(e)}")
            # Continue processing even if markdown save fails
        
        # Extract store details
        logger.info("Extracting store details...")
        store_details = extract_store_details(extracted_content)
        logger.info(f"Extracted store details: {store_details}")
        
        # Get location and weather data
        location_data = get_location_from_ip()
        store_details["City"] = location_data["City"]
        store_details["Country"] = location_data["country"]
        
        # Get and store weather data, get reference ID
        weather_details = get_weather(location_data["City"])
        weather_id = store_weather_data(weather_details)
        
        # Parse table data
        logger.info("Parsing table data...")
        df = parse_markdown_table(extracted_content)
        
        if df is not None:
            logger.info(f"Successfully parsed table with {len(df)} rows")
            # Merge store data and add weather reference
            merged_df = merge_medical_store_data(df, weather_id)
            
            # Add store details as columns to every row in the DataFrame
            for key, value in store_details.items():
                # Convert snake_case keys to Title Case for consistency
                column_key = ' '.join(word.capitalize() for word in key.split('_'))
                if column_key in ['Doctor Name', 'Pharmacy Name', 'Hospital Name', 'Lab Name']:
                    merged_df[column_key] = value
            # Enrich with master data references
            enriched_df = enrich_ocr_data(merged_df)
            
            # Store in MongoDB
            mongo_data = enriched_df.to_dict(orient="records")
            result = collection_ocr.insert_many(mongo_data)
            
            ## Update metadata collection with new document IDs and markdown file reference
            metadata_update = {
                "$push": {
                    "ocr_document_ids": {
                        "$each": [str(id) for id in result.inserted_ids]
                    }
                },
                "$set": {
                    "last_updated": datetime.datetime.now(),
                    "markdown_file": markdown_path if os.path.exists(markdown_path) else None
                },
                "$inc": {
                    "total_documents": len(result.inserted_ids)
                }
            }
            collection_metadata.update_one(
                {"collection_name": "ocr_data"},
                metadata_update,
                upsert=True
            )

            end_time = time.time()
            total_time = end_time - start_time
            print(f"Document processed in {total_time:.2f} seconds")
            print(f"Enriched data saved to MongoDB ({collection_ocr.name})")
            print(f"Weather data saved to MongoDB ({collection_weather.name})")
            print(f"Metadata updated in database_3")
            
            if os.path.exists(markdown_path):
                print(f"Markdown saved to: {markdown_path}")
            else:
                print("Warning: Failed to save markdown file")
            
            return enriched_df
        else:
            logger.warning("Failed to parse table data from document")
            return None
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}", exc_info=True)
        return None

def process_existing_ocr_data():
    """Process existing OCR data in the database to enrich it with master data using bulk operations."""
    start_time = time.time()
    
    # Fetch weather data once
    city = get_location_from_ip()["City"]
    weather_details = get_weather(city)
    weather_id = store_weather_data(weather_details)
    
    # Fetch all OCR records at once
    ocr_records = list(collection_ocr.find({"weather_id": {"$exists": False}}))
    if not ocr_records:
        print("No OCR records found that need updating")
        return
    
    print(f"Processing {len(ocr_records)} OCR records...")
    
    # Gather unique values for batch processing
    doctor_names = set()
    pharmacy_names = set()
    hospital_names = set()
    lab_names = set()
    medicine_names = set()
    
    for record in ocr_records:
        if record.get("Doctor Name"):
            doctor_names.add(record["Doctor Name"])
        if record.get("Pharmacy Name"):
            pharmacy_names.add(record["Pharmacy Name"])
        if record.get("Hospital Name"):
            hospital_names.add(record["Hospital Name"]) 
        if record.get("Lab Name"):
            lab_names.add(record["Lab Name"])
        if record.get("Particular"):
            medicine_names.add(record["Particular"])
    
    # Batch fetch master data
    doctors_lookup = {}
    if doctor_names:
        doctors_cursor = collection_doctor.find({"doctor_name": {"$in": list(doctor_names)}})
        doctors_lookup = {d["doctor_name"]: d for d in doctors_cursor}
    
    pharmacies_lookup = {}
    if pharmacy_names:
        pharmacies_cursor = collection_pharmacy.find({"pharmacy_name": {"$in": list(pharmacy_names)}})
        pharmacies_lookup = {p["pharmacy_name"]: p for p in pharmacies_cursor}
    
    hospitals_lookup = {}
    if hospital_names:
        hospitals_cursor = collection_hospital.find({"hospital_name": {"$in": list(hospital_names)}})
        hospitals_lookup = {h["hospital_name"]: h for h in hospitals_cursor}
    
    labs_lookup = {}
    if lab_names:
        labs_cursor = collection_lab.find({"lab_name": {"$in": list(lab_names)}})
        labs_lookup = {l["lab_name"]: l for l in labs_cursor}
    
    # Medicine lookup for batch
    medicine_lookup = match_medicine_batch(list(medicine_names))
    
    # Prepare bulk update operations
    bulk_updates = []
    for record in ocr_records:
        # Add timestamp for existing records that don't have it
        current_time = datetime.datetime.now()
        update_data = {
            "weather_id": weather_id,
            "timestamp": record.get("timestamp", current_time),
            "created_date": record.get("created_date", current_time.strftime('%Y-%m-%d')),
            "created_time": record.get("created_time", current_time.strftime('%H:%M:%S')),
            "last_updated": current_time
        }
        
        # Set fields to unset (weather data fields that should be referenced)
        fields_to_unset = {}
        for weather_field in ["Temperature", "Humidity", "Wind Speed", "Condition", 
                             "Wind Direction", "Pressure", "Visibility", 
                             "Cloud Cover", "UV Index", "Precipitation"]:
            if weather_field in record:
                fields_to_unset[weather_field] = ""
        
        # Match from pre-fetched lookups
        if record.get("Doctor Name") and record["Doctor Name"] in doctors_lookup:
            update_data["doctor_id"] = doctors_lookup[record["Doctor Name"]]["_id"]
            
        if record.get("Pharmacy Name") and record["Pharmacy Name"] in pharmacies_lookup:
            update_data["pharmacy_id"] = pharmacies_lookup[record["Pharmacy Name"]]["_id"]
            
        if record.get("Hospital Name") and record["Hospital Name"] in hospitals_lookup:
            update_data["hospital_id"] = hospitals_lookup[record["Hospital Name"]]["_id"]
            
        if record.get("Lab Name") and record["Lab Name"] in labs_lookup:
            update_data["lab_id"] = labs_lookup[record["Lab Name"]]["_id"]
        
        # Medicine lookup with fallback to partial matching
        medicine_name = record.get("Particular", "").strip().lower()
        if medicine_name:
            # Try exact match first
            matched_med = medicine_lookup.get(medicine_name)
            
            # If not found, try to find the closest match
            if not matched_med:
                for key in medicine_lookup:
                    if medicine_name in key or key in medicine_name:
                        matched_med = medicine_lookup[key]
                        break
            
            if matched_med:
                update_data["medicine_id"] = matched_med["_id"]
        
        # Create update operation
        update_op = {"$set": update_data}
        if fields_to_unset:
            update_op["$unset"] = fields_to_unset
        
        # Add to bulk update
        bulk_updates.append(UpdateOne({"_id": record["_id"]}, update_op))
    
    # Execute bulk update if there are updates to make
    if bulk_updates:
        result = collection_ocr.bulk_write(bulk_updates)
        end_time = time.time()
        print(f"Updated {result.modified_count} OCR records with master details")
        print(f"Processing completed in {end_time - start_time:.2f} seconds")

def get_weather_for_user(user_location):
    """Get weather based on location provided by the user's browser."""
    city = user_location.get('city', '')
    if not city:
        # Fall back to IP-based location if user location not available
        location_data = get_location_from_ip()
        city = location_data['City']  # Note: our implementation uses 'City' with capital C
    
    # Get weather data for the city
    weather_data = get_weather(city)
    return weather_data

def save_purchase_data(form_data, document_id=None):
    """Save purchase invoice data to the database."""
    try:
        # Extract form data
        pharmacy_name = form_data.get('pharmacy_name', '')
        address = form_data.get('address', '')
        phone = form_data.get('phone', '')
        email = form_data.get('email', '')
        patient_name = form_data.get('patient_name', '')
        patient_age = form_data.get('patient_age', '')
        
        # Extract medicine details - these are arrays
        # Handle both Flask request.form and dictionary inputs
        if hasattr(form_data, 'getlist'):
            medicine_names = form_data.getlist('medicine_name[]')
            manufacturers = form_data.getlist('manufacturer[]')
            packagings = form_data.getlist('packaging[]')
            quantities = form_data.getlist('quantity[]')
            rates = form_data.getlist('rate[]')
            discounts = form_data.getlist('discount[]')
        else:
            # Convert to lists if they're not already
            medicine_names = form_data.get('medicine_name[]', [])
            if not isinstance(medicine_names, list):
                medicine_names = [medicine_names] if medicine_names else []
                
            manufacturers = form_data.get('manufacturer[]', [])
            if not isinstance(manufacturers, list):
                manufacturers = [manufacturers] if manufacturers else []
                
            packagings = form_data.get('packaging[]', [])
            if not isinstance(packagings, list):
                packagings = [packagings] if packagings else []
                
            quantities = form_data.get('quantity[]', [])
            if not isinstance(quantities, list):
                quantities = [quantities] if quantities else []
                
            rates = form_data.get('rate[]', [])
            if not isinstance(rates, list):
                rates = [rates] if rates else []
                
            discounts = form_data.get('discount[]', [])
            if not isinstance(discounts, list):
                discounts = [discounts] if discounts else []
        
        # Ensure we have at least one item
        if not medicine_names:
            medicine_names = ['']
        
        # Ensure all arrays have the same length
        max_length = max(len(medicine_names), len(manufacturers), len(packagings), 
                         len(quantities), len(rates), len(discounts))
        
        # Pad arrays to ensure they all have the same length
        medicine_names = medicine_names + [''] * (max_length - len(medicine_names))
        manufacturers = manufacturers + [''] * (max_length - len(manufacturers))
        packagings = packagings + [''] * (max_length - len(packagings))
        quantities = quantities + ['1'] * (max_length - len(quantities))
        rates = rates + ['0'] * (max_length - len(rates))
        discounts = discounts + ['0'] * (max_length - len(discounts))
        
        # Get invoice settings with safe conversion
        try:
            overall_discount = float(form_data.get('overall_discount', 0))
        except (ValueError, TypeError):
            overall_discount = 0
            
        try:
            gst_rate = float(form_data.get('gst_rate', 12))
        except (ValueError, TypeError):
            gst_rate = 12
        
        # Calculate totals
        items = []
        subtotal = 0
        total_discount = 0
        
        for i in range(max_length):
            try:
                # Get values with safe indexing
                medicine_name = medicine_names[i] if i < len(medicine_names) else ''
                manufacturer = manufacturers[i] if i < len(manufacturers) else ''
                packaging = packagings[i] if i < len(packagings) else ''
                
                # Skip empty items
                if not medicine_name.strip():
                    continue
                
                # Safe conversion with fallbacks
                try:
                    quantity = int(quantities[i]) if i < len(quantities) and quantities[i] else 1
                except (ValueError, TypeError):
                    quantity = 1
                    
                try:
                    rate = float(rates[i]) if i < len(rates) and rates[i] else 0
                except (ValueError, TypeError):
                    rate = 0
                    
                try:
                    discount = float(discounts[i]) if i < len(discounts) and discounts[i] else 0
                except (ValueError, TypeError):
                    discount = 0
                
                # Calculate item amount
                amount = quantity * rate
                discount_amount = amount * (discount / 100)
                taxable_amount = amount - discount_amount
                
                # Add to totals
                subtotal += taxable_amount
                total_discount += discount_amount
                
                # Add item to list
                items.append({
                    'medicine_name': medicine_name,
                    'manufacturer': manufacturer,
                    'packaging': packaging,
                    'quantity': quantity,
                    'rate': rate,
                    'discount': discount,
                    'amount': amount,
                    'discount_amount': discount_amount,
                    'taxable_amount': taxable_amount
                })
            except Exception as e:
                logger.warning(f"Error processing item {i}: {e}")
                continue
        
        # Apply overall discount
        overall_discount_amount = subtotal * (overall_discount / 100)
        subtotal_after_discount = subtotal - overall_discount_amount
        total_discount += overall_discount_amount
        
        # Calculate GST
        gst_amount = subtotal_after_discount * (gst_rate / 100)
        grand_total = subtotal_after_discount + gst_amount
        
        # Create invoice number
        invoice_number = f"INV-{int(time.time())}"
        
        # Create invoice data object
        invoice_data = {
            'pharmacy_name': pharmacy_name,
            'address': address,
            'phone': phone,
            'email': email,
            'patient_name': patient_name,
            'patient_age': patient_age,
            'date': datetime.datetime.now().strftime('%Y-%m-%d'),
            'invoice_number': invoice_number,
            'items': items,
            'subtotal': round(subtotal_after_discount, 2),
            'total_discount': round(total_discount, 2),
            'gst_rate': gst_rate,
            'gst_amount': round(gst_amount, 2),
            'grand_total': round(grand_total, 2),
            'document_id': document_id,
            'timestamp': datetime.datetime.now()
        }
        
        # Save to database
        result = collection_purchase.insert_one(invoice_data)
        logger.info(f"Saved purchase data with ID: {result.inserted_id}")
        
        return invoice_data
        
    except Exception as e:
        logger.error(f"Error saving purchase data: {e}")
        # Add stack trace for better debugging
        import traceback
        logger.error(traceback.format_exc())
        return None

# Update other routes to require login
def login_required(route_function):
    @wraps(route_function)
    def wrapper(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return route_function(*args, **kwargs)
    return wrapper

@app.route('/profile')
@login_required
def profile():
    """Display user profile information and upload history."""
    user = collection_enduser.find_one({'username': session['username']})
    if user:
        # Get user's bills and patient bills
        bills = get_user_bills(session['username'])
        
        # Get total statistics
        total_uploads = len(bills)
        total_amount = sum(float(bill.get('amount', 0)) for bill in bills)
        
        # Get recent activity (last 5 uploads)
        recent_activity = sorted(bills, key=lambda x: x['upload_time'], reverse=True)[:5]
        
        # Convert string dates to datetime objects
        for bill in recent_activity:
            if isinstance(bill.get('upload_time'), str):
                try:
                    bill['upload_time'] = datetime.datetime.fromisoformat(bill['upload_time'].replace('Z', '+00:00'))
                except (ValueError, TypeError):
                    # If conversion fails, use current time as fallback
                    bill['upload_time'] = datetime.datetime.now()
        
        # Get patient statistics
        patients = get_user_patients(session['username'])
        patient_count = len(patients)
        
        # Ensure user's created_at is a datetime object
        if isinstance(user.get('created_at'), str):
            try:
                user['created_at'] = datetime.datetime.fromisoformat(user['created_at'].replace('Z', '+00:00'))
            except (ValueError, TypeError):
                user['created_at'] = datetime.datetime.now()
        
        return render_template('profile.html', 
                             user=user,
                             total_uploads=total_uploads,
                             total_amount=total_amount,
                             recent_activity=recent_activity,
                             patient_count=patient_count,
                             patients=patients)  # Add patients to template context
    return redirect(url_for('login'))

@app.route('/', methods=['GET', 'POST'])
@login_required
def home():
    # Get user data and location/weather info
    user = collection_enduser.find_one({'username': session['username']})
    location_data = get_location_from_ip()
    weather_data = get_weather(location_data["City"])
    
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return render_template('index.html',
                                location=location_data,
                                weather=weather_data,
                                user=user)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'error')
            return render_template('index.html',
                                location=location_data,
                                weather=weather_data,
                                user=user)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            static_upload_folder = os.path.join(app.root_path, 'static', 'uploads')
            os.makedirs(static_upload_folder, exist_ok=True)
            filepath = os.path.join(static_upload_folder, filename)
            file.save(filepath)
            logger.info(f"Processing file: {filename}")

            # Save image path in session for later deletion and template rendering
            session['uploaded_image_path'] = filepath
            image_url = url_for('static', filename=f'uploads/{filename}')
            
            try:
                # Convert document and extract content
                logger.info("Starting document conversion...")
                accelerator_options = AcceleratorOptions(num_threads=16, device=AcceleratorDevice.CUDA)
                pipeline_options = PdfPipelineOptions()
                pipeline_options.accelerator_options = accelerator_options
                pipeline_options.do_ocr = True
                pipeline_options.do_table_structure = True
                pipeline_options.table_structure_options.do_cell_matching = True

                converter = DocumentConverter(
                    allowed_formats=[InputFormat.PDF, InputFormat.IMAGE, InputFormat.CSV],
                    format_options={
                        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
                        InputFormat.DOCX: WordFormatOption(pipeline_cls=SimplePipeline)
                    }
                )

                conversion_result = converter.convert(filepath)
                extracted_content = conversion_result.document.export_to_markdown()
                logger.info(f"Extracted content length: {len(extracted_content)}")
                
                # Save markdown file
                try:
                    markdown_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'markdown')
                    os.makedirs(markdown_dir, exist_ok=True)
                    
                    input_filename = os.path.splitext(filename)[0]
                    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                    markdown_filename = f"{input_filename}_{timestamp}.md"
                    markdown_path = os.path.join(markdown_dir, markdown_filename)
                    
                    logger.info(f"Attempting to save markdown to: {markdown_path}")
                    with open(markdown_path, "w", encoding="utf-8") as md_file:
                        md_file.write(extracted_content)
                        md_file.flush()
                    
                    if os.path.exists(markdown_path):
                        file_size = os.path.getsize(markdown_path)
                        logger.info(f"Successfully saved markdown file. Size: {file_size} bytes")
                    else:
                        logger.error(f"Failed to save markdown file at: {markdown_path}")
                        
                except Exception as e:
                    logger.error(f"Error while saving markdown file: {str(e)}")
                    # Continue processing even if markdown save fails
                
                # Extract store details
                logger.info("Extracting store details...")
                store_details = extract_store_details(extracted_content)
                logger.info(f"Extracted store details: {store_details}")
                
                # Parse table data
                logger.info("Parsing table data...")
                df = parse_markdown_table(extracted_content)
                
                if df is not None and not df.empty:
                    logger.info(f"Successfully parsed table with {len(df)} rows")
                    # Store the extracted data in session for later use
                    session['extracted_content'] = extracted_content
                    session['store_details'] = store_details

                    # Add store details to DataFrame for display
                    for key, value in store_details.items():
                        # Convert snake_case keys to Title Case for consistency
                        column_key = ' '.join(word.capitalize() for word in key.split('_'))
                        if column_key in ['Doctor Name', 'Pharmacy Name', 'Hospital Name', 'Lab Name']:
                            df[column_key] = value

                    return render_template('index.html', 
                                        store_details=store_details,
                                        extracted_data=df,
                                        show_edit_form=True,
                                        location=location_data,
                                        weather=weather_data,
                                        user=user,
                                        markdown_path=markdown_path if os.path.exists(markdown_path) else None,
                                        uploaded_image_url = image_url)  # Pass image URL to template
                else:
                    logger.warning("No table data found in the document")
                    flash("No table data found in the document. Please check the file format.", "error")
                    return render_template('index.html',
                                        location=location_data,
                                        weather=weather_data,
                                        user=user,
                                        uploaded_image_url = image_url)
                    
            except Exception as e:
                logger.error(f"Error processing document: {str(e)}", exc_info=True)
                flash(f"Error processing document: {str(e)}", "error")
                return render_template('index.html',
                                    location=location_data,
                                    weather=weather_data,
                                    user=user,
                                    uploaded_image_url = image_url)
            finally:
                pass
        else:
            flash("File type not allowed", "error")
            return render_template('index.html',
                                location=location_data,
                                weather=weather_data,
                                user=user,
                                uploaded_image_url = session.get('uploaded_image_path'))
    
    # GET request - show the main page
    return render_template('index.html',
                         location=location_data,
                         weather=weather_data,
                         user=user)

@app.route('/confirm', methods=['POST'])
def confirm_data():
    try:
        store_details_json = request.form.get('store_details_json')
        table_data_json = request.form.get('table_data_json')
        
        if not store_details_json or not table_data_json:
            return jsonify({
                'success': False,
                'error': 'Missing required data'
            }), 400

        store_details = json.loads(store_details_json)
        table_data = json.loads(table_data_json)
        
        # Convert table data to DataFrame
        df = pd.DataFrame(table_data)
        # Add store details as columns to DataFrame
        for key, value in store_details.items():
            # Convert snake_case keys to Title Case for consistency
            column_key = ' '.join(word.capitalize() for word in key.split('_'))
            if column_key in ['Doctor Name', 'Pharmacy Name', 'Hospital Name', 'Lab Name']:
                df[column_key] = value
                
        # Enrich the data
        enriched_data = enrich_ocr_data(df)
        
        # Get user data from Flask session
        user_data = {
            'username': session.get('username'),
            'email': session.get('email')
        }
        
        # Store data with relationship handling
        success, message = store_ocr_data_with_relationships(
            user_data=user_data,
            store_details=store_details,
            enriched_data=enriched_data
        )

        # Delete the uploaded image after processing
        uploaded_image_path = session.pop('uploaded_image_path', None)
        if uploaded_image_path and os.path.exists(uploaded_image_path):
            os.remove(uploaded_image_path)

        if success:
            return render_template(
                'result.html',
                enriched_data=enriched_data,
                store_details=store_details,
                message="Data processed and stored successfully!"
            )
        else:
            return jsonify({
                'success': False,
                'error': message
            }), 500
            
    except Exception as e:
        logger.error(f"Error in confirm_data: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/set-user-location', methods=['POST'])
def set_user_location():
    """Handle location updates from the browser."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400

        # Validate required fields
        required_fields = ['city', 'state', 'country', 'latitude', 'longitude']
        missing_fields = [field for field in required_fields if not data.get(field)]
        if missing_fields:
            return jsonify({
                'success': False,
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400

        is_manual_update = data.get('isManualUpdate', False)
        
        # Get current location and weather from session
        current_location = session.get('user_location', {})
        current_weather_id = session.get('weather_id')
        
        # Create new location data with proper validation
        try:
            latitude = float(data['latitude'])
            longitude = float(data['longitude'])
            if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
                raise ValueError("Invalid coordinates")
        except (ValueError, TypeError):
            return jsonify({
                'success': False,
                'error': 'Invalid coordinates provided'
            }), 400

        location_data = {
            'City': data['city'].strip(),
            'country': data['country'].strip(),
            'state': data['state'].strip(),
            'area': data.get('area', '').strip(),
            'lat': latitude,
            'lon': longitude,
            'source': data.get('source', 'browser'),
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # Check if location has significantly changed (more than 100 meters)
        def haversine_distance(lat1, lon1, lat2, lon2):
            from math import radians, sin, cos, sqrt, atan2
            R = 6371000  # Earth's radius in meters
            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * atan2(sqrt(a), sqrt(1-a))
            return R * c

        location_changed = True
        if current_location:
            try:
                distance = haversine_distance(
                    float(current_location.get('lat', 0)),
                    float(current_location.get('lon', 0)),
                    latitude,
                    longitude
                )
                location_changed = (
                    distance > 100 or  # More than 100 meters
                    current_location.get('City') != location_data['City'] or
                    current_location.get('state') != location_data['state']
                )
            except (ValueError, TypeError):
                # If there's any error in calculation, assume location changed
                location_changed = True
        
        weather_changed = False
        weather_data = None
        
        # Update weather if location changed or manual update
        if location_changed or is_manual_update:
            # Get new weather data
            weather_data = get_weather(location_data['City'])
            if weather_data:
                try:
                    weather_id = store_weather_data(weather_data)
                    if weather_id:
                        weather_changed = str(weather_id) != str(current_weather_id)
                        session['weather_id'] = str(weather_id)
                except Exception as e:
                    logger.error(f"Error storing weather data: {e}")
                    # Continue even if weather storage fails
                    pass
            
            # Update session with new location
            session['user_location'] = location_data
            
            # If user is logged in, update their location in the database
            if 'username' in session:
                try:
                    collection_enduser.update_one(
                        {'username': session['username']},
                        {
                            '$set': {
                                'location': {
                                    'type': 'Point',
                                    'coordinates': [longitude, latitude]
                                },
                                'city': location_data['City'],
                                'state': location_data['state'],
                                'last_location_update': datetime.datetime.now()
                            }
                        }
                    )
                except Exception as e:
                    logger.error(f"Error updating user location in database: {e}")
        
        return jsonify({
            'success': True,
            'location': location_data,
            'locationChanged': location_changed,
            'weatherChanged': weather_changed,
            'weather': weather_data,
            'message': 'Location updated successfully' if (location_changed or weather_changed) else 'Location is up to date'
        })
        
    except Exception as e:
        logger.error(f"Error setting user location: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username_or_email = request.form.get('username_or_email')
        password = request.form.get('password')
        
        if not username_or_email or not password:
            flash('Please provide both username/email and password', 'error')
            return render_template('login.html')
        
        user = authenticate_user(username_or_email, password)
        if user:
            # Update last login time
            collection_enduser.update_one(
                {'username': user['username']},
                {'$set': {'last_login': datetime.datetime.now()}}
            )
            
            # Set session variables
            session['username'] = user['username']
            session['email'] = user['email']
            session['full_name'] = user['full_name']
            
            flash('Logged in successfully!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid username/email or password', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # Get form data
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        full_name = request.form.get('full_name')
        mobile = request.form.get('mobile')
        city = request.form.get('city')
        state = request.form.get('state')
        latitude = request.form.get('latitude')
        longitude = request.form.get('longitude')
        
        try:
            # Validate passwords match
            if password != confirm_password:
                raise ValueError("Passwords do not match")
                
            # Validate password strength
            is_valid, password_errors = validate_password(password)
            if not is_valid:
                raise ValueError("\n".join(password_errors))
            
            # Create user data dictionary
            user_data = {
                'username': username,
                'email': email,
                'password': password,
                'full_name': full_name,
                'mobile': mobile,
                'city': city,
                'state': state,
                'location': {
                    'type': 'Point',
                    'coordinates': [float(longitude), float(latitude)]
                } if latitude and longitude else None
            }
            
            # Create user
            username = create_user(user_data)
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
            
        except ValueError as e:
            flash(str(e), 'error')
        except Exception as e:
            flash('Error creating user. Please try again.', 'error')
            logger.error(f"Error in register route: {e}")
    
    return render_template('register.html')

@app.route('/debug/users', methods=['GET'])
def debug_users():
    """Debug endpoint to check user database."""
    try:
        # Check if user is authenticated and is admin (add your admin check logic)
        if 'user_id' not in session:
            return jsonify({"error": "Not authenticated"}), 401
            
        # Get all users from the database
        users = list(collection_enduser.find({}, {"password": 0}))  # Exclude passwords
        
        # Convert ObjectId to string for JSON serialization
        for user in users:
            if '_id' in user:
                user['_id'] = str(user['_id'])
                
        # Get database connection info
        db_info = {
            "connection": str(client),
            "database": db_master.name,
            "collection": collection_enduser.name,
            "user_count": len(users)
        }
        
        return jsonify({
            "db_info": db_info,
            "users": users
        })
    except Exception as e:
        logger.error(f"Error in debug users route: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/debug/connection', methods=['GET'])
def debug_connection():
    """Debug endpoint to check MongoDB connection details."""
    try:
        # Get MongoDB connection info
        connection_info = {
            "address": f"{client.address[0]}:{client.address[1]}",
            "is_primary": client.is_primary,
            "server_info": client.server_info(),
            "databases": client.list_database_names()
        }
        
        return jsonify({
            "connection_info": connection_info,
            "message": "This endpoint shows the current MongoDB connection details."
        })
    except Exception as e:
        logger.error(f"Error in debug connection route: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully', 'success')
    return redirect(url_for('login'))

@app.route('/check-email', methods=['POST'])
def check_email():
    """API endpoint to check email availability."""
    email = request.json.get('email', '')
    if not email:
        return jsonify({'available': False, 'error': 'Email is required'})
    
    if not is_valid_email(email):
        return jsonify({
            'available': False, 
            'error': 'Invalid email format'
        })
    
    is_available = check_email_availability(email)
    return jsonify({'available': is_available})

@app.route('/check-username', methods=['POST'])
def check_username():
    """API endpoint to check username availability."""
    username = request.json.get('username', '')
    if not username:
        return jsonify({'available': False, 'error': 'Username is required'})
    
    if not is_valid_username(username):
        return jsonify({
            'available': False, 
            'error': 'Invalid username format. Use 3-16 characters, letters, numbers, underscore, or hyphen.'
        })
    
    is_available = check_username_availability(username)
    return jsonify({'available': is_available})

@app.route('/patients')
@login_required
def view_patients():
    patients = get_user_patients(session['username'])
    return render_template('patients.html', patients=patients)

@app.route('/bills')
@login_required
def view_bills():
    """
    Display all bills for the logged-in user, including both personal and patient bills.
    """
    try:
        username = session.get('username')
        if not username:
            flash('Please log in to view bills.', 'error')
            return redirect(url_for('login'))
        
        bills = get_user_bills(username)
        
        return render_template('bills.html', bills=bills)
    except Exception as e:
        logger.error(f"Error in view_bills route: {e}")
        flash('An error occurred while fetching bills.', 'error')
        return redirect(url_for('home'))

# Add this route after other routes
@app.route('/delete-patient/<patient_id>', methods=['DELETE'])
@login_required
def delete_patient(patient_id):
    try:
        username = session.get('username')
        if not username:
            return jsonify({'success': False, 'error': 'Not authenticated'}), 401

        # First get the patient data to find associated bills
        patient_doc = collection_patient.find_one(
            {"_id": username, "patients": {"$elemMatch": {"patient_id": patient_id}}}
        )
        
        if not patient_doc:
            return jsonify({'success': False, 'error': 'Patient not found'}), 404

        # Find the specific patient in the array
        patient = next((p for p in patient_doc.get('patients', []) if p.get('patient_id') == patient_id), None)
        if not patient:
            return jsonify({'success': False, 'error': 'Patient not found'}), 404

        # Get all bill IDs associated with this patient
        bill_ids = [bill.get('_id') for bill in patient.get('bills', [])]

        # Delete all OCR data records associated with this patient's bills
        if bill_ids:
            collection_ocr.delete_many({"_id": {"$in": bill_ids}})

        # Remove the patient and their bills from master_patient collection
        result = collection_patient.update_one(
            {"_id": username},
            {"$pull": {"patients": {"patient_id": patient_id}}}
        )

        if result.modified_count > 0:
            # Update patient count in master_enduser
            updated_patient_doc = collection_patient.find_one({"_id": username})
            current_patient_count = len(updated_patient_doc.get('patients', [])) if updated_patient_doc else 0
            
            collection_enduser.update_one(
                {"username": username},
                {
                    "$set": {
                        "patient_count": current_patient_count,
                        "last_activity": datetime.datetime.now()
                    }
                }
            )
            
            return jsonify({
                'success': True, 
                'message': 'Patient and associated data deleted successfully'
            })
        else:
            return jsonify({'success': False, 'error': 'Failed to delete patient'}), 500

    except Exception as e:
        logger.error(f"Error deleting patient: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Add these routes after other routes
@app.route('/edit-profile', methods=['GET', 'POST'])
@login_required
def edit_profile():
    user = collection_enduser.find_one({'username': session['username']})
    if not user:
        flash('User not found.', 'error')
        return redirect(url_for('login'))

    if request.method == 'POST':
        try:
            # Get form data
            full_name = request.form.get('full_name')
            email = request.form.get('email')
            mobile = request.form.get('mobile')
            city = request.form.get('city')
            state = request.form.get('state')
            latitude = request.form.get('latitude')
            longitude = request.form.get('longitude')
            current_password = request.form.get('current_password')
            new_password = request.form.get('new_password')
            confirm_password = request.form.get('confirm_password')

            # Validate required fields
            if not all([full_name, email, mobile]):
                flash('Please fill in all required fields.', 'error')
                return render_template('edit_profile.html', user=user)

            # Validate email format
            is_valid, error = validate_email(email)
            if not is_valid:
                flash(error, 'error')
                return render_template('edit_profile.html', user=user)

            # Validate mobile format
            is_valid, error = validate_phone_number(mobile)
            if not is_valid:
                flash(error, 'error')
                return render_template('edit_profile.html', user=user)

            # Check if email is changed and already exists
            if email != user['email'] and not check_email_availability(email):
                flash('This email is already registered.', 'error')
                return render_template('edit_profile.html', user=user)

            # Prepare update data
            update_data = {
                'full_name': full_name,
                'email': email,
                'mobile': mobile,
                'city': city,
                'state': state,
                'last_updated': datetime.datetime.now()
            }

            # Update location if provided
            if latitude and longitude:
                try:
                    lat = float(latitude)
                    lon = float(longitude)
                    if -90 <= lat <= 90 and -180 <= lon <= 180:
                        update_data['location'] = {
                            'type': 'Point',
                            'coordinates': [lon, lat]
                        }
                except ValueError:
                    pass

            # Handle password change if requested
            if current_password and new_password and confirm_password:
                if not check_password_hash(user['password'], current_password):
                    flash('Current password is incorrect.', 'error')
                    return render_template('edit_profile.html', user=user)

                if new_password != confirm_password:
                    flash('New passwords do not match.', 'error')
                    return render_template('edit_profile.html', user=user)

                # Validate password strength
                is_valid, password_errors = validate_password(new_password)
                if not is_valid:
                    flash("\n".join(password_errors), 'error')
                    return render_template('edit_profile.html', user=user)

                update_data['password'] = generate_password_hash(new_password)

            # Update user in database
            result = collection_enduser.update_one(
                {'username': session['username']},
                {'$set': update_data}
            )

            if result.modified_count > 0:
                flash('Profile updated successfully!', 'success')
                if email != session.get('email'):
                    session['email'] = email
                return redirect(url_for('profile'))
            else:
                flash('No changes were made.', 'info')
                return render_template('edit_profile.html', user=user)

        except Exception as e:
            logger.error(f"Error updating profile: {e}")
            flash('An error occurred while updating your profile.', 'error')
            return render_template('edit_profile.html', user=user)

    return render_template('edit_profile.html', user=user)

@app.route('/purchase', methods=['GET', 'POST'])
@login_required
def purchase():
    """Handle purchase page and form submission."""
    if 'username' not in session:
        flash('Please log in to access this page', 'warning')
        return redirect(url_for('login'))
        
    if request.method == 'POST':
        try:
            # Get form data
            file = request.files.get('invoice')
            
            if not file or file.filename == '':
                flash('No file selected', 'error')
                return redirect(request.url)
                
            if not allowed_file(file.filename):
                flash('Invalid file type. Please upload PDF, PNG, JPG, JPEG, or CSV', 'error')
                return redirect(request.url)
                
            # Save file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Extract content from document
            try:
                # Convert document and extract content
                logger.info(f"Converting document: {file_path}")
                accelerator_options = AcceleratorOptions(num_threads=16, device=AcceleratorDevice.CUDA)
                pipeline_options = PdfPipelineOptions()
                pipeline_options.accelerator_options = accelerator_options
                pipeline_options.do_ocr = True
                pipeline_options.do_table_structure = True
                pipeline_options.table_structure_options.do_cell_matching = True

                converter = DocumentConverter(
                    allowed_formats=[InputFormat.PDF, InputFormat.IMAGE, InputFormat.CSV],
                    format_options={
                        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
                        InputFormat.DOCX: WordFormatOption(pipeline_cls=SimplePipeline)
                    }
                )

                conversion_result = converter.convert(file_path)
                extracted_content = conversion_result.document.export_to_markdown()
                logger.info(f"Extracted content length: {len(extracted_content)}")
                
                # Save markdown file
                markdown_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'markdown')
                os.makedirs(markdown_dir, exist_ok=True)
                
                input_filename = os.path.splitext(filename)[0]
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                markdown_filename = f"{input_filename}_{timestamp}.md"
                markdown_path = os.path.join(markdown_dir, markdown_filename)
                
                with open(markdown_path, "w", encoding="utf-8") as md_file:
                    md_file.write(extracted_content)
                
                # Extract store details
                store_details = extract_store_details(extracted_content)
                
                # Parse table data from markdown
                df = parse_markdown_table(extracted_content)
                
                if df is not None and not df.empty:
                    # Generate a unique document ID
                    document_id = str(uuid.uuid4())
                    
                    # Store the extracted data in session for later use
                    session['temp_extracted_data'] = {
                        'document_id': document_id,
                        'extracted_content': extracted_content,
                        'store_details': store_details,
                        'markdown_path': markdown_path,
                        'file_path': file_path,
                        'extracted_data': df.to_dict('records'),
                        'timestamp': datetime.datetime.now().isoformat()
                    }
                    
                    # Convert DataFrame to HTML table for display
                    table_html = df.to_html(classes='table table-striped table-bordered', index=False)
                    
                    return render_template('purchase.html', 
                                          store_details=store_details,
                                          table_html=table_html,
                                          extracted_data=df.to_dict('records'),
                                          document_id=document_id,
                                          show_edit_form=True)
                else:
                    flash("No table data found in the document. Please check the file format.", "error")
                    return redirect(request.url)
                    
            except Exception as e:
                logger.error(f"Error extracting content: {str(e)}", exc_info=True)
                flash(f"Error extracting content: {str(e)}", "error")
                return redirect(request.url)
            finally:
                # We'll keep the file until processing is complete
                pass
                
        except Exception as e:
            flash(f'Error: {str(e)}', 'error')
            return redirect(request.url)
            
    # GET request - show purchase form
    return render_template('purchase.html',
                          location=session.get('user_location'),
                          weather=session.get('weather_data'))

@app.route('/confirm-purchase', methods=['POST'])
@login_required
def confirm_purchase():
    """Handle the form submission to store purchase data."""
    try:
        # Get the document ID from the form
        document_id = request.form.get('document_id')
        
        # Get the temporary extracted data from the session
        temp_data = session.get('temp_extracted_data', {})
        
        if not temp_data or temp_data.get('document_id') != document_id:
            flash('Session expired or invalid document ID', 'danger')
            return redirect(url_for('purchase'))
        
        # Get form data for store details
        store_details = {
            'pharmacy_name': request.form.get('pharmacy_name', ''),
            'address': request.form.get('address', ''),
            'phone_number': request.form.get('phone', ''),
            'email': request.form.get('email', ''),
            'patient_name': request.form.get('patient_name', ''),
            'patient_age': request.form.get('patient_age', '')
        }
        
        # Get table data from form
        medicine_names = request.form.getlist('medicine_name[]')
        quantities = request.form.getlist('quantity[]')
        rates = request.form.getlist('rate[]')
        amounts = request.form.getlist('amount[]')
        
        # Create table data
        table_data = []
        for i in range(len(medicine_names)):
            if medicine_names[i].strip():  # Only process non-empty medicine names
                try:
                    quantity = int(quantities[i]) if quantities[i] else 0
                    rate = float(rates[i]) if rates[i] else 0.0
                    amount = float(amounts[i]) if amounts[i] else 0.0
                except (ValueError, TypeError):
                    quantity = 0
                    rate = 0.0
                    amount = 0.0
                
                table_data.append({
                    'medicine_name': medicine_names[i],
                    'quantity': quantity,
                    'rate': rate,
                    'amount': amount
                })
        
        # Store in database_2.purchase_data only
        purchase_data = {
            'document_id': document_id,
            'user_id': session.get('user_id'),
            'username': session.get('username'),
            'store_details': store_details,
            'table_data': table_data,
            'original_extracted_data': temp_data.get('extracted_data'),
            'markdown_path': temp_data.get('markdown_path'),
            'timestamp': datetime.datetime.now(),
            'location': session.get('user_location')
        }
        
        # Insert into the purchase_data collection
        result = collection_purchase.insert_one(purchase_data)
        
        # Clean up files if needed
        file_path = temp_data.get('file_path')
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
        
        # Clear the temporary data from the session
        if 'temp_extracted_data' in session:
            del session['temp_extracted_data']
        
        flash('Purchase data saved successfully!', 'success')
        return redirect(url_for('dashboard'))
                              
    except Exception as e:
        logger.error(f"Error saving purchase data: {e}")
        flash(f"Error saving purchase data: {str(e)}", 'danger')
        return redirect(url_for('purchase'))

if __name__ == '__main__':
    try:
        setup_database_indices()
        app.run(debug=True, host='0.0.0.0', port=5001)
    except Exception as e:
        print(f"Failed to start application: {e}")