"""
Delhi Outdoor Safety Prediction - Label Creation
Step 2: Creating YES/NO safety labels based on health thresholds
"""
# Copyright (c) 2026 Praveen Kumar. All Rights Reserved.

import pandas as pd
import numpy as np

class SafetyLabelCreator:
    """
    Creates outdoor safety labels (YES/NO) based on environmental conditions
    Uses WHO and Indian health standards
    """
    
    def __init__(self):
        """
        Define health-based thresholds for each parameter
        These are based on WHO and Indian air quality standards
        """
        # Air Quality Thresholds (WHO guidelines)
        self.thresholds = {
            'pm25_unsafe': 60,      # PM2.5 > 60 μg/m³ is unsafe
            'pm10_unsafe': 100,     # PM10 > 100 μg/m³ is unsafe
            'aqi_unsafe': 150,      # AQI > 150 is unhealthy
            'co_unsafe': 10000,     # CO > 10000 μg/m³ is unsafe
            'no2_unsafe': 200,      # NO2 > 200 μg/m³ is unsafe
            'so2_unsafe': 80,       # SO2 > 80 μg/m³ is unsafe
            'o3_unsafe': 180,       # O3 > 180 μg/m³ is unsafe
            'temp_high': 40,        # Temperature > 40°C is uncomfortable
            'temp_low': 5,          # Temperature < 5°C is uncomfortable
            'wind_speed_low': 1.0,  # Low wind = poor dispersion
            'humidity_high': 90     # Very high humidity is uncomfortable
        }
    
    def is_safe_conditions(self, row):
        """
        Determine if conditions are safe for outdoor activity
        Returns: 1 (YES - Safe) or 0 (NO - Unsafe)
        """
        unsafe_reasons = []
        
        # Check PM2.5 (most important)
        if pd.notna(row['pm25']) and row['pm25'] > self.thresholds['pm25_unsafe']:
            unsafe_reasons.append(f"PM2.5 too high ({row['pm25']:.1f})")
        
        # Check PM10
        if pd.notna(row['pm10']) and row['pm10'] > self.thresholds['pm10_unsafe']:
            unsafe_reasons.append(f"PM10 too high ({row['pm10']:.1f})")
        
        # Check AQI
        if pd.notna(row['aqi']) and row['aqi'] > self.thresholds['aqi_unsafe']:
            unsafe_reasons.append(f"AQI too high ({row['aqi']:.0f})")
        
        # Check Carbon Monoxide
        if pd.notna(row['co']) and row['co'] > self.thresholds['co_unsafe']:
            unsafe_reasons.append(f"CO too high ({row['co']:.0f})")
        
        # Check Nitrogen Dioxide
        if pd.notna(row['no2']) and row['no2'] > self.thresholds['no2_unsafe']:
            unsafe_reasons.append(f"NO2 too high ({row['no2']:.1f})")
        
        # Check Sulfur Dioxide
        if pd.notna(row['so2']) and row['so2'] > self.thresholds['so2_unsafe']:
            unsafe_reasons.append(f"SO2 too high ({row['so2']:.1f})")
        
        # Check Ozone
        if pd.notna(row['o3']) and row['o3'] > self.thresholds['o3_unsafe']:
            unsafe_reasons.append(f"Ozone too high ({row['o3']:.1f})")
        
        # Check Temperature extremes
        if pd.notna(row['temperature']):
            if row['temperature'] > self.thresholds['temp_high']:
                unsafe_reasons.append(f"Temperature too high ({row['temperature']:.1f}C)")
            elif row['temperature'] < self.thresholds['temp_low']:
                unsafe_reasons.append(f"Temperature too low ({row['temperature']:.1f}C)")
        
        # Check for stagnant air (low wind + high pollution)
        if (pd.notna(row['wind_speed']) and 
            row['wind_speed'] < self.thresholds['wind_speed_low'] and
            pd.notna(row['pm25']) and row['pm25'] > 35):
            unsafe_reasons.append("Poor air dispersion (low wind + pollution)")
        
        # Decision: Safe if NO unsafe reasons found
        is_safe = len(unsafe_reasons) == 0
        
        return {
            'safe': 1 if is_safe else 0,
            'safe_label': 'YES' if is_safe else 'NO',
            'reasons': unsafe_reasons if not is_safe else ['All parameters within safe limits']
        }
    
    def create_labels(self, df):
        """
        Apply safety labels to entire dataset
        """
        print("\n[INFO] Creating safety labels...")
        print("=" * 60)
        
        # Apply safety check to each row
        results = df.apply(self.is_safe_conditions, axis=1)
        
        # Extract results
        df['outdoor_safe'] = [r['safe'] for r in results]
        df['outdoor_safe_label'] = [r['safe_label'] for r in results]
        df['safety_reasons'] = ['; '.join(r['reasons']) for r in results]
        
        # Print statistics
        safe_count = df['outdoor_safe'].sum()
        total_count = len(df)
        unsafe_count = total_count - safe_count
        
        print(f"\n[INFO] Label Distribution:")
        print(f"  Total samples: {total_count}")
        print(f"  Safe (YES): {safe_count} ({safe_count/total_count*100:.1f}%)")
        print(f"  Unsafe (NO): {unsafe_count} ({unsafe_count/total_count*100:.1f}%)")
        
        # Show sample unsafe days
        print(f"\n[INFO] Sample Unsafe Conditions:")
        unsafe_samples = df[df['outdoor_safe'] == 0].head(3)
        for idx, row in unsafe_samples.iterrows():
            print(f"\n  Sample {idx}:")
            print(f"    Date: {row['timestamp']}")
            print(f"    Reasons: {row['safety_reasons']}")
        
        return df
    
    def print_threshold_info(self):
        """
        Display the thresholds being used
        """
        print("\n[INFO] Health-Based Safety Thresholds:")
        print("=" * 60)
        print("\nAir Quality Limits:")
        print(f"  PM2.5: <= {self.thresholds['pm25_unsafe']} ug/m3")
        print(f"  PM10: <= {self.thresholds['pm10_unsafe']} ug/m3")
        print(f"  AQI: <= {self.thresholds['aqi_unsafe']}")
        print(f"  CO: <= {self.thresholds['co_unsafe']} ug/m3")
        print(f"  NO2: <= {self.thresholds['no2_unsafe']} ug/m3")
        print(f"  SO2: <= {self.thresholds['so2_unsafe']} ug/m3")
        print(f"  O3: <= {self.thresholds['o3_unsafe']} ug/m3")
        
        print("\nComfort Limits:")
        print(f"  Temperature: {self.thresholds['temp_low']}C - {self.thresholds['temp_high']}C")
        print(f"  Wind Speed: >= {self.thresholds['wind_speed_low']} m/s (for good dispersion)")
        print(f"  Humidity: <= {self.thresholds['humidity_high']}%")


# ====================
# HOW TO USE THIS CODE
# ====================

if __name__ == "__main__":
    print("=" * 60)
    print("Delhi Outdoor Safety - Label Creation")
    print("=" * 60)
    
    # STEP 1: Load your collected data
    # Replace with your actual filename
    try:
        df = pd.read_csv('delhi_environmental_data_20260106.csv')
        print(f"\n[SUCCESS] Data loaded: {len(df)} samples")
        print(f"Columns: {list(df.columns)}")
    except FileNotFoundError:
        print("\n[WARNING] Data file not found!")
        print("Please run the data collection script first.")
        exit()
    
    # STEP 2: Create label creator
    label_creator = SafetyLabelCreator()
    
    # STEP 3: Show thresholds being used
    label_creator.print_threshold_info()
    
    # STEP 4: Create labels
    df_labeled = label_creator.create_labels(df)
    
    # STEP 5: Save labeled data
    output_filename = 'delhi_data_with_labels.csv'
    df_labeled.to_csv(output_filename, index=False)
    print(f"\n[SUCCESS] Labeled data saved to: {output_filename}")
    
    # STEP 6: Show sample data
    print("\n[INFO] Sample Labeled Data:")
    print(df_labeled[['timestamp', 'temperature', 'pm25', 'aqi', 
                      'outdoor_safe_label', 'safety_reasons']].head(10))