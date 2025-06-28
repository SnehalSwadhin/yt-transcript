# in database.py
import sqlite3

DB_NAME = "car_data.db"

def initialize_db():
    conn = sqlite3.connect(DB_NAME)
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
        dealer_location TEXT
    )
    ''')
    # Table to track which videos have been processed
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS processed_videos (
        video_id TEXT PRIMARY KEY
    )
    ''')
    conn.commit()
    conn.close()

def is_video_processed(video_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT video_id FROM processed_videos WHERE video_id = ?", (video_id,))
    result = cursor.fetchone()
    conn.close()
    return result is not None

def add_cars_to_db(car_list):
    # car_list is the list of dicts from your LLM
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    for car in car_list:
        cursor.execute('''
        INSERT INTO cars (timestamp, video_id, published_at, car_oem, model, variant, price, colour, odometer, year, service_record, frame_type, transmission_type, fuel_type, num_owners, rto, city, engine_details, feature_details, rating, start_timestamp, video_link, video_title, channel_title, dealer_name, dealer_contact, dealer_email, dealer_website, dealer_location)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            car.get('timestamp', None),
            car.get('video_id', None),
            car.get('published_at', None),
            car.get('OEM', None),
            car.get('Model', None),
            car.get('Variant', None),
            car.get('Price', None),
            car.get('Colour', None),
            car.get('Odometer', None),
            car.get('Year', None),
            car.get('Service Record', None),
            car.get('Frame Type', None),
            car.get('Transmission Type', None),
            car.get('Fuel Type', None),
            car.get('Number of owners', None),
            car.get('RTO', None),
            car.get('City', None),
            car.get('Engine Details', None),
            car.get('Feature Details', None),
            car.get('Rating', None),
            car.get('start_timestamp', None),
            car.get('video_link', None),
            car.get('video_title', None),
            car.get('channel_title', None),
            car.get('dealer_name', None),
            car.get('dealer_contact', None),
            car.get('dealer_email', None),
            car.get('dealer_website', None),
            car.get('dealer_location', None)
        ))
    # Mark the video as processed
    if car_list:
        video_id = car_list[0].get('video_id')
        cursor.execute("INSERT OR IGNORE INTO processed_videos (video_id) VALUES (?)", (video_id,))
    conn.commit()
    conn.close()