# in database.py
import sqlite3
import libsql
import re
import numpy as np
import os
import dotenv
from logger import log

dotenv.load_dotenv()

DB_URL = os.environ.get("TURSO_DATABASE_URL")
DB_TOKEN = os.environ.get("TURSO_AUTH_TOKEN")


def initialize_db():
    log.info("Initializing the database...")
    conn = libsql.connect("local.db", sync_url=DB_URL, auth_token=DB_TOKEN)
    log.info("Connected to the database successfully.")
    cursor = conn.cursor()

    # Table to store scraped car details
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS cars (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        video_id TEXT NOT NULL,
        published_at TEXT,
        car_oem TEXT,
        model TEXT,
        variant TEXT,
        price TEXT,
        price_numeric REAL,
        colour TEXT,
        odometer TEXT,
        year TEXT,
        service_record TEXT,
        frame_type TEXT,
        transmission_type TEXT,
        fuel_type TEXT,
        num_owners TEXT,
        rto TEXT,
        city TEXT,
        engine_details TEXT,
        feature_details TEXT,
        rating TEXT,
        start_timestamp TEXT,
        video_link TEXT,
        video_title TEXT,
        channel_title TEXT,
        dealer_name TEXT,
        dealer_contact TEXT,
        dealer_email TEXT,
        dealer_website TEXT,
        dealer_location TEXT,
        original_quote TEXT
    )
    ''')
    # Table to track which videos have been processed
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS processed_videos (
        video_id TEXT PRIMARY KEY
    )
    ''')
    conn.commit()
    conn.sync()
    conn.close()

def is_video_processed(video_id):
    conn = libsql.connect("local.db", sync_url=DB_URL, auth_token=DB_TOKEN)
    cursor = conn.cursor()
    cursor.execute("SELECT video_id FROM processed_videos WHERE video_id = ?", (video_id,))
    result = cursor.fetchone()
    conn.close()
    return result is not None

def parse_price(price_str):
    """
    Parses a complex price string into a float number.
    Handles 'lakh', commas, currency symbols, and other text.
    """
    # Ensure input is a string and convert to lowercase
    try:
        price_str = str(price_str).lower()
    except:
        return np.nan # Return NaN for non-string types

    if '%' in price_str or 'discount' in price_str or 'request' in price_str:
        return np.nan
    
    if '/' in price_str:
        price_str = price_str.split('/')[0]
    
    if '-' in price_str:
        price_str = price_str.split('/')[0]

    # Case 1: The string contains "lakh"
    if 'lakh' in price_str:
        # Use regex to find the number (can be float) before "lakh"
        # \d+ matches digits, \. matches a literal dot. ([\d.]+) captures it.
        match = re.search(r'([\d.]+)', price_str)
        if match:
            value = float(match.group(1))
            return value * 100000  # 1 lakh = 100,000
        else:
            return np.nan # No number found before "lakh"

    # Case 2: Standard number format (with commas, etc.)
    else:
        # Use regex to remove all non-digit characters
        # [^\d] means "any character that is NOT a digit".
        numeric_part = re.sub(r'[^\d]', '', price_str)
        if numeric_part:
            return float(numeric_part)
        else:
            return np.nan # Return NaN if no digits are found

def add_cars_to_db(car_list):
    # car_list is the list of dicts from your LLM
    conn = libsql.connect("local.db", sync_url=DB_URL, auth_token=DB_TOKEN)
    cursor = conn.cursor()
    for car in car_list:
        price_raw = car.get('Price', None)
        price_numeric = parse_price(price_raw)
        # Set price_numeric to 0 if it's NaN or None
        if np.isnan(price_numeric) or price_numeric is None:
            price_numeric = 0
        cursor.execute('''
        INSERT INTO cars (timestamp, video_id, published_at, car_oem, model, variant, price, price_numeric, colour, odometer, year, service_record, frame_type, transmission_type, fuel_type, num_owners, rto, city, engine_details, feature_details, rating, start_timestamp, video_link, video_title, channel_title, dealer_name, dealer_contact, dealer_email, dealer_website, dealer_location, original_quote)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            car.get('timestamp', ''),
            car.get('video_id', ''),
            car.get('published_at', ''),
            car.get('OEM', ''),
            car.get('Model', ''),
            car.get('Variant', ''),
            price_raw or '',
            price_numeric,
            car.get('Colour', ''),
            car.get('Odometer', ''),
            car.get('Year', ''),
            car.get('Service Record', ''),
            car.get('Frame Type', ''),
            car.get('Transmission Type', ''),
            car.get('Fuel Type', ''),
            car.get('Number of owners', ''),
            car.get('RTO', ''),
            car.get('City', ''),
            car.get('Engine Details', ''),
            car.get('Feature Details', ''),
            car.get('Rating', ''),
            car.get('start_timestamp', ''),
            car.get('video_link', ''),
            car.get('video_title', ''),
            car.get('channel_title', ''),
            car.get('dealer_name', ''),
            car.get('dealer_contact', ''),
            car.get('dealer_email', ''),
            car.get('dealer_website', ''),
            car.get('dealer_location', ''),
            car.get('original_quote', '')
        ))
        
    # Mark the video as processed
    if car_list:
        video_id = car_list[0].get('video_id')
        cursor.execute("INSERT OR IGNORE INTO processed_videos (video_id) VALUES (?)", (video_id,))
    conn.commit()
    conn.sync()
    conn.close()