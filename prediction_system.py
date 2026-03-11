"""
Delhi Outdoor Safety - Real-Time Prediction System
Step 4: Making predictions with live data
"""
# Copyright (c) 2026 Praveen Kumar. All Rights Reserved.

import joblib
import pandas as pd
import requests
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class RealTimeSafetyPredictor:
    """
    Real-time prediction system for outdoor safety
    """
    
    def __init__(self, model_path, openweather_api_key):
        """
        Load the trained model and setup API connection
        """
        print("🔄 Loading trained model...")
        
        # Load model package
        self.model_package = joblib.load(model_path)
        self.model = self.model_package['model']
        self.scaler = self.model_package['scaler']
        self.feature_names = self.model_package['feature_names']
        self.model_name = self.model_package['model_name']
        
        # API setup
        self.api_key = openweather_api_key
        self.delhi_lat = 28.6139
        self.delhi_lon = 77.2090
        
        print(f"✅ Model loaded: {self.model_name}")
        print(f"   Features: {len(self.feature_names)}")
    
    def get_current_data(self):
        """
        Fetch current environmental data from APIs
        """
        print("\n🌍 Fetching live environmental data for Delhi...")
        
        data = {}
        
        # Get weather data
        try:
            weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={self.delhi_lat}&lon={self.delhi_lon}&appid={self.api_key}&units=metric"
            weather_response = requests.get(weather_url, timeout=10)
            weather_data = weather_response.json()
            
            data['temperature'] = weather_data['main']['temp']
            data['humidity'] = weather_data['main']['humidity']
            data['pressure'] = weather_data['main']['pressure']
            data['wind_speed'] = weather_data['wind']['speed']
            data['rainfall'] = weather_data.get('rain', {}).get('1h', 0)
            
            print("  ✓ Weather data collected")
        except Exception as e:
            print(f"  ✗ Weather data error: {e}")
            return None
        
        # Get air quality data
        try:
            aq_url = "https://api.openaq.org/v2/latest?limit=100&country=IN&city=Delhi"
            aq_response = requests.get(aq_url, timeout=10)
            aq_data = aq_response.json()
            
            # Extract pollutants
            pollutants = {'pm25': None, 'pm10': None, 'co': None, 
                         'no2': None, 'so2': None, 'o3': None}
            
            if 'results' in aq_data:
                for result in aq_data['results']:
                    for measurement in result.get('measurements', []):
                        param = measurement['parameter'].lower()
                        if param in pollutants:
                            pollutants[param] = measurement['value']
            
            data.update(pollutants)
            
            # Calculate AQI
            if data['pm25'] is not None:
                data['aqi'] = self._calculate_aqi(data['pm25'], data['pm10'])
            else:
                data['aqi'] = None
            
            print("  ✓ Air quality data collected")
        except Exception as e:
            print(f"  ✗ Air quality data error: {e}")
            return None
        
        return data
    
    def _calculate_aqi(self, pm25, pm10):
        """Calculate simplified AQI"""
        if pm25 is not None:
            if pm25 <= 30:
                return 50
            elif pm25 <= 60:
                return 100
            elif pm25 <= 90:
                return 150
            elif pm25 <= 120:
                return 200
            elif pm25 <= 250:
                return 300
            else:
                return 400
        return 100
    
    def predict_tomorrow_safety(self):
        """
        Predict if it's safe to go outdoor tomorrow
        """
        print("\n" + "=" * 70)
        print("DELHI OUTDOOR SAFETY PREDICTION FOR TOMORROW")
        print("=" * 70)
        
        # Get current data
        current_data = self.get_current_data()
        
        if current_data is None:
            print("\n❌ Unable to fetch data. Please try again.")
            return None
        
        # Display current conditions
        print("\n📊 Current Environmental Conditions:")
        print(f"   Temperature: {current_data['temperature']:.1f}°C")
        print(f"   Humidity: {current_data['humidity']:.0f}%")
        print(f"   Wind Speed: {current_data['wind_speed']:.1f} m/s")
        if current_data['pm25']:
            print(f"   PM2.5: {current_data['pm25']:.1f} μg/m³")
        if current_data['aqi']:
            print(f"   AQI: {current_data['aqi']:.0f}")
        
        # Prepare for prediction
        df_predict = pd.DataFrame([current_data])
        
        # Ensure all required features are present
        for feat in self.feature_names:
            if feat not in df_predict.columns:
                df_predict[feat] = None
        
        # Select only model features
        X_predict = df_predict[self.feature_names]
        
        # Fill missing values with median (or 0 for first prediction)
        X_predict = X_predict.fillna(0)
        
        # Scale features
        X_predict_scaled = self.scaler.transform(X_predict)
        
        # Make prediction
        prediction = self.model.predict(X_predict_scaled)[0]
        probability = self.model.predict_proba(X_predict_scaled)[0]
        
        # Display prediction
        print("\n" + "=" * 70)
        if prediction == 1:
            print("✅ PREDICTION: YES - SAFE TO GO OUTDOORS TOMORROW")
            print(f"   Confidence: {probability[1]*100:.1f}%")
            print("\n   The environmental conditions are expected to be within safe limits.")
            print("   General outdoor activities should be fine for healthy individuals.")
        else:
            print("⚠️  PREDICTION: NO - OUTDOOR EXPOSURE NOT RECOMMENDED TOMORROW")
            print(f"   Confidence: {probability[0]*100:.1f}%")
            print("\n   Air quality or weather conditions may pose health risks.")
            print("   Consider the following:")
            print("   • Limit outdoor physical activities")
            print("   • Wear N95 mask if you must go outside")
            print("   • Keep windows closed")
            print("   • Monitor AQI throughout the day")
        
        print("=" * 70)
        
        # Health recommendations
        self._print_health_recommendations(current_data, prediction)
        
        return {
            'prediction': 'SAFE' if prediction == 1 else 'UNSAFE',
            'confidence': probability[prediction] * 100,
            'data': current_data,
            'timestamp': datetime.now()
        }
    
    def _print_health_recommendations(self, data, prediction):
        """
        Print health recommendations based on conditions
        """
        print("\n💡 Health Recommendations:")
        print("-" * 70)
        
        if prediction == 0:
            # Unsafe conditions
            if data.get('pm25') and data['pm25'] > 60:
                print("   🔴 High PM2.5: Wear N95 mask, avoid outdoor exercise")
            
            if data.get('aqi') and data['aqi'] > 200:
                print("   🔴 Poor AQI: Stay indoors, use air purifiers")
            
            if data.get('temperature') and data['temperature'] > 38:
                print("   🔴 High Temperature: Stay hydrated, avoid midday sun")
            
            print("   • Vulnerable groups (children, elderly, respiratory patients)")
            print("     should stay indoors")
            print("   • If symptoms develop (cough, breathlessness), seek medical help")
        else:
            # Safe conditions
            print("   ✅ Good conditions for outdoor activities")
            print("   • Morning walks and exercise recommended")
            print("   • Still maintain general precautions for pollution-prone areas")
        
        print("-" * 70)


# ====================
# HOW TO USE THIS CODE
# ====================

if __name__ == "__main__":
    import sys
    
    print("=" * 70)
    print("DELHI OUTDOOR SAFETY - REAL-TIME PREDICTION SYSTEM")
    print("=" * 70)
    
    # STEP 1: Enter your API key
    API_KEY = "YOUR_API_KEY_HERE"  # Replace with your actual API key
    MODEL_PATH = "delhi_safety_model.pkl"
    
    if API_KEY == "YOUR_API_KEY_HERE":
        print("\n⚠️ Please enter your OpenWeather API key!")
        sys.exit()
    
    # STEP 2: Check if model exists
    try:
        # Load model
        predictor = RealTimeSafetyPredictor(MODEL_PATH, API_KEY)
        
        # STEP 3: Make prediction
        result = predictor.predict_tomorrow_safety()
        
        if result:
            print(f"\n📅 Prediction made at: {result['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Model used: {predictor.model_name}")
        
        # STEP 4: Option to run continuously
        print("\n" + "=" * 70)
        print("💡 TIP: Run this script daily to get updated predictions!")
        print("=" * 70)
        
    except FileNotFoundError:
        print(f"\n❌ Model file not found: {MODEL_PATH}")
        print("Please train the model first using the ML training script.")
    except Exception as e:
        print(f"\n❌ Error: {e}")