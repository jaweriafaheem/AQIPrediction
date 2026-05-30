import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def calculate_aqi_pm25(pm25):
    """US EPA PM2.5 AQI Formula"""
    if pd.isna(pm25) or pm25 < 0:
        return np.nan
    elif pm25 <= 12.0:
        return round((50 - 0) / (12.0 - 0) * (pm25 - 0) + 0)
    elif pm25 <= 35.4:
        return round((100 - 51) / (35.4 - 12.1) * (pm25 - 12.1) + 51)
    elif pm25 <= 55.4:
        return round((150 - 101) / (55.4 - 35.5) * (pm25 - 35.5) + 101)
    elif pm25 <= 150.4:
        return round((200 - 151) / (150.4 - 55.5) * (pm25 - 55.5) + 151)
    elif pm25 <= 250.4:
        return round((300 - 201) / (250.4 - 150.5) * (pm25 - 150.5) + 201)
    elif pm25 <= 350.4:
        return round((400 - 301) / (350.4 - 250.5) * (pm25 - 250.5) + 301)
    elif pm25 <= 500.4:
        return round((500 - 401) / (500.4 - 350.5) * (pm25 - 350.5) + 401)
    else:
        return 500

def calculate_aqi_pm10(pm10):
    """US EPA PM10 AQI Formula"""
    if pd.isna(pm10) or pm10 < 0:
        return np.nan
    elif pm10 <= 54:
        return round((50 - 0) / (54 - 0) * (pm10 - 0) + 0)
    elif pm10 <= 154:
        return round((100 - 51) / (154 - 55) * (pm10 - 55) + 51)
    elif pm10 <= 254:
        return round((150 - 101) / (254 - 155) * (pm10 - 155) + 101)
    elif pm10 <= 354:
        return round((200 - 151) / (354 - 255) * (pm10 - 255) + 151)
    elif pm10 <= 424:
        return round((300 - 201) / (424 - 355) * (pm10 - 355) + 201)
    elif pm10 <= 504:
        return round((400 - 301) / (504 - 425) * (pm10 - 425) + 301)
    elif pm10 <= 604:
        return round((500 - 401) / (604 - 505) * (pm10 - 505) + 401)
    else:
        return 500

def fetch_data():
    print("Fetching historical data for Karachi...")
    lat = 24.8607
    lon = 67.0011
    
    # Yesterday's date
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    start_date = "2024-01-01"
    
    # 1. Fetch Air Quality data from Open-Meteo
    print(f"1. Fetching air quality data from {start_date} to {yesterday}...")
    aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    aq_params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "pm2_5,pm10,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide,ozone",
        "start_date": start_date,
        "end_date": yesterday,
        "timezone": "Asia/Karachi"
    }
    aq_res = requests.get(aq_url, params=aq_params).json()
    
    # 2. Fetch Meteorological Weather data from Open-Meteo Archive API
    print(f"2. Fetching weather archive data from {start_date} to {yesterday}...")
    weather_url = "https://archive-api.open-meteo.com/v1/archive"
    weather_params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": yesterday,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        "timezone": "Asia/Karachi"
    }
    w_res = requests.get(weather_url, params=weather_params).json()
    
    # Parse to pandas
    df_aq = pd.DataFrame(aq_res["hourly"])
    df_w = pd.DataFrame(w_res["hourly"])
    
    # Merge on time
    df = pd.merge(df_aq, df_w, on="time")
    df["time"] = pd.to_datetime(df["time"])
    df["date"] = df["time"].dt.strftime("%Y-%m-%d")
    
    # Aggregate daily averages
    print("3. Aggregating to daily statistics...")
    daily_df = df.groupby("date").agg({
        "pm2_5": "mean",
        "pm10": "mean",
        "nitrogen_dioxide": "mean",
        "sulphur_dioxide": "mean",
        "carbon_monoxide": "mean",
        "ozone": "mean",
        "temperature_2m": "mean",
        "relative_humidity_2m": "mean",
        "wind_speed_10m": "mean"
    }).reset_index()
    
    # Rename columns for clarity and consistency
    daily_df.rename(columns={
        "temperature_2m": "temp",
        "relative_humidity_2m": "humidity",
        "wind_speed_10m": "wind_speed"
    }, inplace=True)
    
    # Compute AQI today
    print("4. Calculating AQI values...")
    daily_df["aqi_pm25"] = daily_df["pm2_5"].apply(calculate_aqi_pm25)
    daily_df["aqi_pm10"] = daily_df["pm10"].apply(calculate_aqi_pm10)
    daily_df["aqi"] = daily_df[["aqi_pm25", "aqi_pm10"]].max(axis=1)
    
    # Add time-based features
    print("5. Generating time features...")
    dates = pd.to_datetime(daily_df["date"])
    daily_df["month"] = dates.dt.month
    daily_df["day"] = dates.dt.day
    daily_df["day_of_week"] = dates.dt.dayofweek
    
    # Add lag features
    print("6. Creating lag features...")
    daily_df["pm2_5_lag_1d"] = daily_df["pm2_5"].shift(1)
    daily_df["pm2_5_lag_2d"] = daily_df["pm2_5"].shift(2)
    daily_df["aqi_lag_1d"] = daily_df["aqi"].shift(1)
    
    # Drop rows with NaN due to lags (first 2 rows)
    daily_df.dropna(subset=["pm2_5_lag_2d", "aqi_lag_1d"], inplace=True)
    
    # Ensure types are correct
    daily_df = daily_df.round(2)
    daily_df["aqi"] = daily_df["aqi"].astype(int)
    daily_df["aqi_lag_1d"] = daily_df["aqi_lag_1d"].astype(int)
    daily_df["month"] = daily_df["month"].astype(int)
    daily_df["day"] = daily_df["day"].astype(int)
    daily_df["day_of_week"] = daily_df["day_of_week"].astype(int)
    
    # Save local copy
    csv_filename = "aqi_historical_data.csv"
    daily_df.to_csv(csv_filename, index=False)
    print(f"Data saved locally as {csv_filename} ({len(daily_df)} rows).")
    return daily_df

def upload_to_hopsworks(df):
    try:
        import os
        if os.path.exists(".env"):
            with open(".env", "r") as f:
                for line in f:
                    if "=" in line:
                        k, v = line.strip().split("=", 1)
                        if v.startswith('"') or v.startswith("'"):
                            v = v[1:-1]
                        os.environ[k] = v
        import hopsworks
        print("\nConnecting to Hopsworks...")
        project = hopsworks.login()
        fs = project.get_feature_store()
        
        print("Registering Feature Group on Hopsworks...")
        # Create/Get Feature Group
        aqi_fg = fs.get_or_create_feature_group(
            name="aqi_karachi_fg",
            version=1,
            primary_key=["date"],
            description="Daily AQI and weather features for Karachi",
            online_enabled=True
        )
        
        aqi_fg.insert(df)
        print("Successfully uploaded historical data to Hopsworks Feature Group!")
    except Exception as e:
        print(f"\nCould not upload to Hopsworks: {e}")
        print("Please check your HOPSWORKS_API_KEY environment variable or network connection.")

if __name__ == "__main__":
    df = fetch_data()
    
    ans = input("\nDo you want to upload this data to Hopsworks now? (y/n): ")
    if ans.lower() == 'y':
        upload_to_hopsworks(df)
    else:
        print("Skipping Hopsworks upload. You can upload later by running this script again.")
