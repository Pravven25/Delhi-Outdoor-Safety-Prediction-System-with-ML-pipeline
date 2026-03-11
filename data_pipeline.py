import time
import requests
import sqlite3
import pandas as pd
from datetime import datetime, timezone
from label_creation import SafetyLabelCreator

# Database configuration
DB_NAME = "delhi_safety_data.db"
TABLE_NAME = "environmental_data"

def init_db():
    \"\"\"Initialize the SQLite database and create the table if it doesn't exist.\"\"\"
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Create a table for our environmental data
    cursor.execute(f'''
    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        temperature REAL,
        humidity REAL,
        pressure REAL,
        wind_speed REAL,
        rainfall REAL,
        pm25 REAL,
        pm10 REAL,
        co REAL,
        no2 REAL,
        so2 REAL,
        o3 REAL,
        aqi REAL,
        safe INTEGER,
        safe_label TEXT,
        reasons TEXT
    )
    ''')
    conn.commit()
    conn.close()
    print(f"✅ Database initialized: {DB_NAME}")

def fetch_and_store_data(api_key):
    \"\"\"Fetch live data, label it, and store it in the database.\"\"\"
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Fetching new data...")
    
    try:
        # 1. Fetch Weather Data (OpenWeather)
        lat, lon = 28.6139, 77.2090
        weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        weather_res = requests.get(weather_url, timeout=10)
        weather_data = weather_res.json()
        
        # 2. Fetch Air Quality Data (Open-Meteo)
        aq_url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone&forecast_days=1"
        aq_res = requests.get(aq_url, timeout=10)
        aq_data = aq_res.json()
        
        current_hour_idx = datetime.now().hour
        hourly = aq_data.get('hourly', {})
        
        # 3. Compile Data
        raw_data = {
            'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S+00:00'),
            'temperature': weather_data['main']['temp'],
            'humidity': weather_data['main']['humidity'],
            'pressure': weather_data['main']['pressure'],
            'wind_speed': weather_data['wind']['speed'],
            'rainfall': weather_data.get('rain', {}).get('1h', 0),
            'pm25': hourly.get('pm2_5', [0])[current_hour_idx] or 0,
            'pm10': hourly.get('pm10', [0])[current_hour_idx] or 0,
            'co': hourly.get('carbon_monoxide', [0])[current_hour_idx] or 0,
            'no2': hourly.get('nitrogen_dioxide', [0])[current_hour_idx] or 0,
            'so2': hourly.get('sulphur_dioxide', [0])[current_hour_idx] or 0,
            'o3': hourly.get('ozone', [0])[current_hour_idx] or 0,
        }
        
        # Calculate AQI (Simplified)
        pm25 = raw_data['pm25']
        if pm25 <= 30: aqi = 50
        elif pm25 <= 60: aqi = 100
        elif pm25 <= 90: aqi = 150
        elif pm25 <= 120: aqi = 200
        elif pm25 <= 250: aqi = 300
        else: aqi = 400
        raw_data['aqi'] = aqi
        
        # 4. Apply Safety Labels 
        # (This is the Data Transformation / ETL step)
        df_new = pd.DataFrame([raw_data])
        creator = SafetyLabelCreator()
        df_labeled = creator.create_labels(df_new)
        
        # 5. Insert into Database (The Load step)
        conn = sqlite3.connect(DB_NAME)
        # We can use pandas to append directly to sqlite!
        df_labeled.to_sql(TABLE_NAME, conn, if_exists='append', index=False)
        conn.close()
        
        print("✅ Successfully ingested new data point into database!")
        
    except Exception as e:
        print(f"❌ Error during data ingestion: {e}")

if __name__ == "__main__":
    print("="*60)
    print(" AUTOMATED DATA INGESTION PIPELINE (ETL)")
    print("="*60)
    
    # Needs your OpenWeather API key
    USER_API_KEY = input("Enter your OpenWeather API Key to start pipeline: ")
    
    init_db()
    
    # Automation Loop
    # In a real enterprise, this would be an Apache Airflow DAG or a Cron Job
    FETCH_INTERVAL_MINUTES = 60
    
    print(f"\n⏳ Pipeline started. Fetching data every {FETCH_INTERVAL_MINUTES} minutes. Press Ctrl+C to stop.")
    
    while True:
        fetch_and_store_data(USER_API_KEY)
        time.sleep(FETCH_INTERVAL_MINUTES * 60)
