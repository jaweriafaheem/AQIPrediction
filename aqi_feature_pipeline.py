import os
import sys
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def calculate_aqi_pm25(pm25):
    if pd.isna(pm25) or pm25 < 0: return np.nan
    elif pm25 <= 12.0: return round((50 - 0) / (12.0 - 0) * (pm25 - 0) + 0)
    elif pm25 <= 35.4: return round((100 - 51) / (35.4 - 12.1) * (pm25 - 12.1) + 51)
    elif pm25 <= 55.4: return round((150 - 101) / (55.4 - 35.5) * (pm25 - 35.5) + 101)
    elif pm25 <= 150.4: return round((200 - 151) / (150.4 - 55.5) * (pm25 - 55.5) + 151)
    elif pm25 <= 250.4: return round((300 - 201) / (250.4 - 150.5) * (pm25 - 150.5) + 201)
    elif pm25 <= 350.4: return round((400 - 301) / (350.4 - 250.5) * (pm25 - 250.5) + 301)
    elif pm25 <= 500.4: return round((500 - 401) / (500.4 - 350.5) * (pm25 - 350.5) + 401)
    else: return 500

def calculate_aqi_pm10(pm10):
    if pd.isna(pm10) or pm10 < 0: return np.nan
    elif pm10 <= 54: return round((50 - 0) / (54 - 0) * (pm10 - 0) + 0)
    elif pm10 <= 154: return round((100 - 51) / (154 - 55) * (pm10 - 55) + 51)
    elif pm10 <= 254: return round((150 - 101) / (254 - 155) * (pm10 - 155) + 101)
    elif pm10 <= 354: return round((200 - 151) / (354 - 255) * (pm10 - 255) + 151)
    elif pm10 <= 424: return round((300 - 201) / (424 - 355) * (pm10 - 355) + 201)
    elif pm10 <= 504: return round((400 - 301) / (504 - 425) * (pm10 - 425) + 301)
    elif pm10 <= 604: return round((500 - 401) / (604 - 505) * (pm10 - 505) + 401)
    else: return 500

def get_aqicn_realtime(token):
    url = f"https://api.waqi.info/feed/karachi/?token={token}"
    try:
        print("Fetching real-time AQI from AQICN API...")
        res = requests.get(url).json()
        if res.get("status") == "ok":
            data = res["data"]
            iaqi = data.get("iaqi", {})
            
            def get_val(key):
                return iaqi.get(key, {}).get("v", np.nan)
            
            record = {
                "pm2_5": get_val("pm25"),
                "pm10": get_val("pm10"),
                "nitrogen_dioxide": get_val("no2"),
                "sulphur_dioxide": get_val("so2"),
                "carbon_monoxide": get_val("co"),
                "ozone": get_val("o3"),
                "temp": get_val("t"),
                "humidity": get_val("h"),
                "wind_speed": get_val("w"),
            }
            print(f"AQICN Real-time Data: PM2.5={record['pm2_5']}, AQI={data.get('aqi')}")
            return record
        else:
            print(f"AQICN returned non-ok status: {res.get('data')}")
            return None
    except Exception as e:
        print(f"Failed to fetch from AQICN: {e}")
        return None

def fetch_features_df(aqicn_token=None):
    lat = 24.8607
    lon = 67.0011
    
    # Define past 7 days up to today
    today_dt = datetime.now()
    start_date = (today_dt - timedelta(days=7)).strftime("%Y-%m-%d")
    end_date = today_dt.strftime("%Y-%m-%d")
    
    print(f"Fetching weather and pollutant trends from Open-Meteo for {start_date} to {end_date}...")
    
    # 1. Fetch Open-Meteo Air Quality (includes forecast/recent history)
    aq_url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    aq_params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "pm2_5,pm10,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide,ozone",
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "Asia/Karachi"
    }
    aq_res = requests.get(aq_url, params=aq_params).json()
    
    # 2. Fetch Open-Meteo Weather Forecast (contains current day weather)
    weather_url = "https://api.open-meteo.com/v1/forecast"
    weather_params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m",
        "timezone": "Asia/Karachi"
    }
    w_res = requests.get(weather_url, params=weather_params).json()
    
    # Parse
    df_aq = pd.DataFrame(aq_res["hourly"])
    df_w = pd.DataFrame(w_res["hourly"])
    
    df = pd.merge(df_aq, df_w, on="time")
    df["time"] = pd.to_datetime(df["time"])
    df["date"] = df["time"].dt.strftime("%Y-%m-%d")
    
    # Group by date for daily averages
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
    
    daily_df.rename(columns={
        "temperature_2m": "temp",
        "relative_humidity_2m": "humidity",
        "wind_speed_10m": "wind_speed"
    }, inplace=True)
    
    # If AQICN token is available, update today's measurements with real-time readings
    today_str = today_dt.strftime("%Y-%m-%d")
    if aqicn_token:
        realtime_data = get_aqicn_realtime(aqicn_token)
        if realtime_data:
            # Find today's row index
            today_idx = daily_df[daily_df["date"] == today_str].index
            if len(today_idx) > 0:
                idx = today_idx[0]
                for col in realtime_data:
                    # Only overwrite if real-time reading is valid (not NaN)
                    if not pd.isna(realtime_data[col]):
                        daily_df.at[idx, col] = realtime_data[col]
                print(f"Updated today's row ({today_str}) with real-time AQICN values.")
    
    # Compute AQI
    daily_df["aqi_pm25"] = daily_df["pm2_5"].apply(calculate_aqi_pm25)
    daily_df["aqi_pm10"] = daily_df["pm10"].apply(calculate_aqi_pm10)
    daily_df["aqi"] = daily_df[["aqi_pm25", "aqi_pm10"]].max(axis=1)
    
    # Compute time components
    dates = pd.to_datetime(daily_df["date"])
    daily_df["month"] = dates.dt.month
    daily_df["day"] = dates.dt.day
    daily_df["day_of_week"] = dates.dt.dayofweek
    
    # Compute lags
    daily_df["pm2_5_lag_1d"] = daily_df["pm2_5"].shift(1)
    daily_df["pm2_5_lag_2d"] = daily_df["pm2_5"].shift(2)
    daily_df["aqi_lag_1d"] = daily_df["aqi"].shift(1)
    
    # Return today's row (which is the last row)
    today_row = daily_df.tail(1).copy()
    today_row = today_row.round(2)
    
    # Cast types
    today_row["aqi"] = today_row["aqi"].astype(int)
    today_row["aqi_lag_1d"] = today_row["aqi_lag_1d"].astype(int)
    today_row["month"] = today_row["month"].astype(int)
    today_row["day"] = today_row["day"].astype(int)
    today_row["day_of_week"] = today_row["day_of_week"].astype(int)
    
    return today_row

def update_local_csv(today_df):
    csv_path = "aqi_historical_data.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        date_val = today_df["date"].values[0]
        
        # Check if today's date already exists
        if date_val in df["date"].values:
            # Overwrite today's row (upsert)
            df.loc[df["date"] == date_val, today_df.columns] = today_df.values
            print(f"Updated existing row for {date_val} in local {csv_path}.")
        else:
            # Append new row
            df = pd.concat([df, today_df], ignore_index=True)
            print(f"Appended new row for {date_val} to local {csv_path}.")
        
        # Save back to CSV
        df.to_csv(csv_path, index=False)
    else:
        print(f"Local {csv_path} not found. Running backfill is recommended first.")

def upload_to_hopsworks(today_df):
    try:
        import hopsworks
        print("\nConnecting to Hopsworks Feature Store...")
        project = hopsworks.login()
        fs = project.get_feature_store()
        aqi_fg = fs.get_feature_group(name="aqi_karachi_fg", version=1)
        
        # Upsert today's row
        aqi_fg.insert(today_df)
        print(f"Successfully upserted today's row to Hopsworks Feature Store!")
    except Exception as e:
        print(f"Failed to upload to Hopsworks: {e}")

def main():
    # Load AQICN token from environment variable if available, else check arguments
    token = os.environ.get("AQICN_TOKEN")
    if not token and os.path.exists(".env"):
        with open(".env", "r") as f:
            for line in f:
                if line.strip().startswith("AQICN_TOKEN="):
                    token = line.split("=", 1)[1].strip()
                    # Remove surrounding quotes if present
                    if token.startswith('"') and token.endswith('"'):
                        token = token[1:-1]
                    elif token.startswith("'") and token.endswith("'"):
                        token = token[1:-1]
    if not token and len(sys.argv) > 1:
        token = sys.argv[1]
        
    if not token:
        print("Warning: No AQICN Token provided. Running solely with Open-Meteo fallbacks.")
        
    today_df = fetch_features_df(token)
    print("\nParsed Features for Today:")
    print(today_df.to_string(index=False))
    
    # Update local emulator CSV
    update_local_csv(today_df)
    
    # Update Hopsworks if key is configured
    if "HOPSWORKS_API_KEY" in os.environ or os.path.exists(os.path.expanduser("~/.hopsworks/key")):
        upload_to_hopsworks(today_df)
    else:
        print("\nHopsworks credentials not detected in env. Skipping Hopsworks upload.")
        print("To upload, set HOPSWORKS_API_KEY environment variable or log in when running backfill.")

if __name__ == "__main__":
    main()
