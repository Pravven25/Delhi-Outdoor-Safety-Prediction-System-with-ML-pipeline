"""
Delhi Outdoor Safety Prediction - Data Collection
Step 1: Collecting Live Environmental Data
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import json
import openmeteo_requests
import requests_cache
from retry_requests import retry
import numpy as np

class DelhiDataCollector:
    """
    This class collects environmental data for Delhi from Open-Meteo and OpenWeather
    Easy to understand for 9th grade students!
    """
    
    def __init__(self, openweather_api_key):
        """
        Initialize with your OpenWeather API key and setup Open-Meteo client
        """
        self.api_key = openweather_api_key
        self.delhi_lat = 28.6139  # Delhi latitude
        self.delhi_lon = 77.2090  # Delhi longitude
        
        # Setup Open-Meteo API client with cache and retry on error
        cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
        retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
        self.openmeteo = openmeteo_requests.Client(session = retry_session)
        
    def get_weather_data(self):
        """
        Get current weather data from OpenWeather
        Returns: temperature, humidity, pressure, wind speed, rainfall
        """
        try:
            url = f"https://api.openweathermap.org/data/2.5/weather?lat={self.delhi_lat}&lon={self.delhi_lon}&appid={self.api_key}&units=metric"
            
            response = requests.get(url, timeout=10)
            data = response.json()
            
            weather_info = {
                'timestamp': datetime.now(),
                'temperature': data['main']['temp'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'wind_speed': data['wind']['speed'],
                'rainfall': data.get('rain', {}).get('1h', 0)  # Last 1 hour rainfall
            }
            
            print("[SUCCESS] Weather data collected successfully!")
            return weather_info
            
        except Exception as e:
            print(f"Error collecting weather data: {e}")
            return None
    
    def get_air_quality_data(self):
        """
        Get air quality data from OpenAQ
        Returns: PM2.5, PM10, CO, NO2, SO2, O3
        """
        try:
            # OpenAQ API endpoint for Delhi
            url = "https://api.openaq.org/v2/latest?limit=100&country=IN&city=Delhi"
            
            response = requests.get(url, timeout=10)
            data = response.json()
            
            # Initialize pollutant dictionary
            pollutants = {
                'pm25': None,
                'pm10': None,
                'co': None,
                'no2': None,
                'so2': None,
                'o3': None
            }
            
            # Extract pollutant values
            if 'results' in data:
                for result in data['results']:
                    for measurement in result.get('measurements', []):
                        param = measurement['parameter'].lower()
                        value = measurement['value']
                        
                        if param == 'pm25':
                            pollutants['pm25'] = value
                        elif param == 'pm10':
                            pollutants['pm10'] = value
                        elif param == 'co':
                            pollutants['co'] = value
                        elif param == 'no2':
                            pollutants['no2'] = value
                        elif param == 'so2':
                            pollutants['so2'] = value
                        elif param == 'o3':
                            pollutants['o3'] = value
            
            print("[SUCCESS] Air quality data collected successfully!")
            return pollutants
            
        except Exception as e:
            print(f"Error collecting air quality data: {e}")
            return None
    
    def calculate_aqi(self, pm25, pm10):
        """
        Calculate simplified AQI based on PM2.5 and PM10
        This is a simplified version for learning purposes
        """
        if pm25 is None and pm10 is None:
            return None
        
        # Use PM2.5 if available, otherwise PM10
        if pm25 is not None:
            if pm25 <= 30:
                return 50  # Good
            elif pm25 <= 60:
                return 100  # Moderate
            elif pm25 <= 90:
                return 150  # Unhealthy for sensitive groups
            elif pm25 <= 120:
                return 200  # Unhealthy
            elif pm25 <= 250:
                return 300  # Very Unhealthy
            else:
                return 400  # Hazardous
        else:
            # Fallback to PM10
            if pm10 <= 50:
                return 50
            elif pm10 <= 100:
                return 100
            elif pm10 <= 250:
                return 150
            elif pm10 <= 350:
                return 200
            else:
                return 300
    
    def collect_complete_data(self):
        """
        Collect both weather and air quality data together for "Live" usage
        Returns a complete data row
        """
        print("\n[INFO] Collecting live data from APIs...")
        
        # Get weather data
        weather = self.get_weather_data()
        
        # Get air quality data
        # We can try to use Open-Meteo for live data too if OpenAQ is failing, 
        # but for now let's keep the existing structure and maybe fallback or use OpenAQ as primary for live
        air_quality = self.get_air_quality_data()
        
        if weather and air_quality:
            # Combine all data
            complete_data = {
                'timestamp': weather['timestamp'],
                'temperature': weather['temperature'],
                'humidity': weather['humidity'],
                'pressure': weather['pressure'],
                'wind_speed': weather['wind_speed'],
                'rainfall': weather['rainfall'],
                'pm25': air_quality['pm25'],
                'pm10': air_quality['pm10'],
                'co': air_quality['co'],
                'no2': air_quality['no2'],
                'so2': air_quality['so2'],
                'o3': air_quality['o3'],
            }
            
            # Calculate AQI
            complete_data['aqi'] = self.calculate_aqi(
                air_quality['pm25'], 
                air_quality['pm10']
            )
            
            return complete_data
        
        return None
    
    def get_historical_aqi_data(self, past_days=14, forecast_days=7):
        """
        Fetch historical and forecast air quality data from Open-Meteo
        Features: PM2.5, PM10, CO, NO2, SO2, O3
        """
        print(f"\n[INFO] Fetching data from Open-Meteo API for past {past_days} days and next {forecast_days} days...")
        
        url = "https://air-quality-api.open-meteo.com/v1/air-quality"
        # Request all required variables
        params = {
            "latitude": self.delhi_lat,
            "longitude": self.delhi_lon,
            "hourly": ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide", "ozone"],
            "past_days": past_days,
            "forecast_days": forecast_days,
        }
        
        try:
            responses = self.openmeteo.weather_api(url, params=params)
            response = responses[0]
            
            print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
            
            # Process hourly data
            hourly = response.Hourly()
            
            # The order matches the "hourly" list in params
            hourly_pm10 = hourly.Variables(0).ValuesAsNumpy()
            hourly_pm2_5 = hourly.Variables(1).ValuesAsNumpy()
            hourly_co = hourly.Variables(2).ValuesAsNumpy()
            hourly_no2 = hourly.Variables(3).ValuesAsNumpy()
            hourly_so2 = hourly.Variables(4).ValuesAsNumpy()
            hourly_o3 = hourly.Variables(5).ValuesAsNumpy()

            hourly_data = {"date": pd.date_range(
                start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
                end =  pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
                freq = pd.Timedelta(seconds = hourly.Interval()),
                inclusive = "left"
            )}
            
            # Map to our project column names
            hourly_data["pm10"] = hourly_pm10
            hourly_data["pm25"] = hourly_pm2_5
            hourly_data["co"] = hourly_co
            hourly_data["no2"] = hourly_no2
            hourly_data["so2"] = hourly_so2
            hourly_data["o3"] = hourly_o3
            
            df = pd.DataFrame(data = hourly_data)
            
            # Rename date to timestamp
            df = df.rename(columns={'date': 'timestamp'})
            
            return df
            
        except Exception as e:
            print(f"Error fetching Open-Meteo data: {e}")
            return None

    def collect_historical_data(self, days=90, samples_per_day=4):
        """
        Collect historical data using Open-Meteo API
        Note: Open-Meteo gives hourly data, so 'samples_per_day' is handled by resampling if needed.
        """
        print(f"\n[INFO] Collecting historical data for {days} days using Open-Meteo...")
        
        # Open-Meteo supports past_days up to 92 for free tier (air quality)
        aqi_df = self.get_historical_aqi_data(past_days=days, forecast_days=0)
        
        if aqi_df is not None:
            print("Fetching matching weather data...")
            
            # Add synthetic weather columns for the project to work
            # (We keep this synthetic part as requested to make it "run good" without full weather API integration yet)
            print("Generating synthetic weather context for historical samples...")
            np.random.seed(42)
            aqi_df['temperature'] = np.random.normal(28, 5, len(aqi_df))  # Mean 28, std 5
            aqi_df['humidity'] = np.random.normal(60, 15, len(aqi_df))
            aqi_df['pressure'] = np.random.normal(1013, 5, len(aqi_df))
            aqi_df['wind_speed'] = np.random.normal(3, 1.5, len(aqi_df)) # m/s
            aqi_df['rainfall'] = np.where(np.random.random(len(aqi_df)) > 0.9, np.random.gamma(2, 2, len(aqi_df)), 0) # Occasional rain
            
            # Calculate AQI (Simplified based on PM2.5/PM10 usually, or we can look up proper formula for all)
            # For this project, we use the class method
            aqi_df['aqi'] = aqi_df.apply(lambda row: self.calculate_aqi(row['pm25'], row['pm10']), axis=1)
            
            # Drop rows with NaN if any
            aqi_df = aqi_df.dropna()
            
            # Ensure specific column order requested by user
            # "Air Pollution Features" ... "ensure the data will be same order"
            # Typical logical order: Timestamp, Pollutants, Weather, AQI or as requested
            # User listed: CO, PM2.5, PM10, NO2, SO2, O3, AQI
            
            desired_order = [
                'timestamp', 
                'co', 'pm25', 'pm10', 'no2', 'so2', 'o3', 'aqi',
                'temperature', 'humidity', 'pressure', 'wind_speed', 'rainfall'
            ]
            
            # Filter to available columns just in case
            cols_to_use = [c for c in desired_order if c in aqi_df.columns]
            aqi_df = aqi_df[cols_to_use]
            
            # Save to CSV
            filename = f'delhi_environmental_data_{datetime.now().strftime("%Y%m%d")}.csv'
            aqi_df.to_csv(filename, index=False)
            print(f"\n[SUCCESS] Data saved to: {filename}")
            print(f"Total samples collected: {len(aqi_df)}")
            
            return aqi_df
            
        return None


# ====================
# HOW TO USE THIS CODE
# ====================

if __name__ == "__main__":
    print("=" * 60)
    print("Delhi Outdoor Safety Prediction - Data Collection")
    print("=" * 60)
    
    # STEP 1: Enter your OpenWeather API key here
    API_KEY = "YOUR_API_KEY_HERE"  # Replace with your actual API key
    
    collector = DelhiDataCollector(API_KEY)

    if API_KEY == "YOUR_API_KEY_HERE":
        print("\n[INFO] OpenWeather API key not found.")
        print("Skipping live weather data test.")
        print("proceeding to test Open-Meteo (Air Quality) which is free...")
        
        # Still create collector for Open-Meteo testing
        # We initialized it above with the dummy key, which is fine for Open-Meteo part
    else:
        # STEP 3: Collect sample data
        print("\n--- Testing Single Data Collection ---")
        sample_data = collector.collect_complete_data()
        
        if sample_data:
            print("\n[SUCCESS] Sample data collected:")
            for key, value in sample_data.items():
                print(f"  {key}: {value}")
    
    # STEP 4: Collect historical data (for demo, 30 days for better model training)
    # This uses Open-Meteo and should work without the OpenWeather key (using synthetic weather)
    print("\n\n--- Collecting Demo Dataset (Open-Meteo) ---")
    try:
        df = collector.collect_historical_data(days=30, samples_per_day=4)
        
        if df is not None:
            print("\n[SUCCESS] Data Preview:")
            print(df.head())
            print(f"\nDataset shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
    except Exception as e:
        print(f"Error executing historical collection: {e}")