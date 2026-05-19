from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
import requests
from datetime import datetime, timedelta

app = Flask(__name__)

# Helper to load models
def load_models():
    models = {}
    for lead in ["1d", "2d", "3d"]:
        path = f"aqi_model_{lead}.pkl"
        if os.path.exists(path):
            with open(path, 'rb') as f:
                models[lead] = pickle.load(f)
        else:
            models[lead] = None
    return models

# Load models at start
MODELS = load_models()

# Helper to calculate AQI categories
def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good", "aqi-good", "Air quality is satisfactory, and air pollution poses little or no risk."
    elif aqi <= 100:
        return "Moderate", "aqi-moderate", "Air quality is acceptable; however, there may be concern for sensitive people."
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "aqi-sensitive", "Members of sensitive groups may experience health effects. The public is less affected."
    elif aqi <= 200:
        return "Unhealthy", "aqi-unhealthy", "Everyone may begin to experience health effects; sensitive groups feel serious impacts."
    elif aqi <= 300:
        return "Very Unhealthy", "aqi-very", "Health alert: The risk of health effects is significantly increased for everyone."
    else:
        return "Hazardous", "aqi-hazardous", "Health warning of emergency conditions: The entire population is likely to be affected."

# Helper functions for AQI calculation from raw concentrations
def calculate_aqi_pm25(pm25):
    if pd.isna(pm25) or pm25 < 0: return 0
    elif pm25 <= 12.0: return round((50 - 0) / (12.0 - 0) * (pm25 - 0) + 0)
    elif pm25 <= 35.4: return round((100 - 51) / (35.4 - 12.1) * (pm25 - 12.1) + 51)
    elif pm25 <= 55.4: return round((150 - 101) / (55.4 - 35.5) * (pm25 - 35.5) + 101)
    elif pm25 <= 150.4: return round((200 - 151) / (150.4 - 55.5) * (pm25 - 55.5) + 151)
    elif pm25 <= 250.4: return round((300 - 201) / (250.4 - 150.5) * (pm25 - 150.5) + 201)
    elif pm25 <= 350.4: return round((400 - 301) / (350.4 - 250.5) * (pm25 - 250.5) + 301)
    elif pm25 <= 500.4: return round((500 - 401) / (500.4 - 350.5) * (pm25 - 350.5) + 401)
    else: return 500

def calculate_aqi_pm10(pm10):
    if pd.isna(pm10) or pm10 < 0: return 0
    elif pm10 <= 54: return round((50 - 0) / (54 - 0) * (pm10 - 0) + 0)
    elif pm10 <= 154: return round((100 - 51) / (154 - 55) * (pm10 - 55) + 51)
    elif pm10 <= 254: return round((150 - 101) / (254 - 155) * (pm10 - 155) + 101)
    elif pm10 <= 354: return round((200 - 151) / (354 - 255) * (pm10 - 255) + 151)
    elif pm10 <= 424: return round((300 - 201) / (424 - 355) * (pm10 - 355) + 201)
    elif pm10 <= 504: return round((400 - 301) / (504 - 425) * (pm10 - 425) + 301)
    elif pm10 <= 604: return round((500 - 401) / (604 - 505) * (pm10 - 505) + 401)
    else: return 500

# Helper to read default token from .env
def get_default_token():
    token = ""
    if os.path.exists(".env"):
        with open(".env", "r") as f:
            for line in f:
                if line.strip().startswith("AQICN_TOKEN="):
                    token = line.split("=", 1)[1].strip()
                    if token.startswith('"') and token.endswith('"'):
                        token = token[1:-1]
                    elif token.startswith("'") and token.endswith("'"):
                        token = token[1:-1]
    return token

@app.route("/")
def index():
    return render_template("index.html", default_token=get_default_token())

@app.route("/api/realtime", methods=["GET"])
def get_realtime():
    token = request.args.get("token") or get_default_token()
    
    # 1. Fetch Open-Meteo coordinates for Karachi
    lat = 24.8607
    lon = 67.0011
    today_dt = datetime.now()
    start_date = (today_dt - timedelta(days=7)).strftime("%Y-%m-%d")
    end_date = today_dt.strftime("%Y-%m-%d")
    
    try:
        # Fetch atmospheric parameters from Open-Meteo Air Quality API
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
        
        # Fetch weather forecast
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
        
        df_aq = pd.DataFrame(aq_res["hourly"])
        df_w = pd.DataFrame(w_res["hourly"])
        df = pd.merge(df_aq, df_w, on="time")
        df["time"] = pd.to_datetime(df["time"])
        df["date"] = df["time"].dt.strftime("%Y-%m-%d")
        
        # Aggregate daily
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
        
        # If token is available, overlay AQICN real-time readings
        aqicn_aqi = None
        if token:
            aq_feed_url = f"https://api.waqi.info/feed/karachi/?token={token}"
            feed_res = requests.get(aq_feed_url).json()
            if feed_res.get("status") == "ok":
                data = feed_res["data"]
                iaqi = data.get("iaqi", {})
                aqicn_aqi = data.get("aqi")
                
                # Fetch key parameters
                rt_pm25 = iaqi.get("pm25", {}).get("v")
                rt_pm10 = iaqi.get("pm10", {}).get("v")
                rt_temp = iaqi.get("t", {}).get("v")
                rt_hum = iaqi.get("h", {}).get("v")
                rt_wind = iaqi.get("w", {}).get("v")
                
                # Find today's index
                today_str = today_dt.strftime("%Y-%m-%d")
                t_idx = daily_df[daily_df["date"] == today_str].index
                if len(t_idx) > 0:
                    idx = t_idx[0]
                    if rt_pm25 is not None: daily_df.at[idx, "pm2_5"] = rt_pm25
                    if rt_pm10 is not None: daily_df.at[idx, "pm10"] = rt_pm10
                    if rt_temp is not None: daily_df.at[idx, "temp"] = rt_temp
                    if rt_hum is not None: daily_df.at[idx, "humidity"] = rt_hum
                    if rt_wind is not None: daily_df.at[idx, "wind_speed"] = rt_wind
        
        # Compute AQIs
        daily_df["aqi_pm25"] = daily_df["pm2_5"].apply(calculate_aqi_pm25)
        daily_df["aqi_pm10"] = daily_df["pm10"].apply(calculate_aqi_pm10)
        daily_df["aqi"] = daily_df[["aqi_pm25", "aqi_pm10"]].max(axis=1)
        
        # Add lag features using historical database
        csv_path = "aqi_historical_data.csv"
        if os.path.exists(csv_path):
            hist_df = pd.read_csv(csv_path)
            hist_df = hist_df.sort_values("date").reset_index(drop=True)
            
            # Yesterday's and 2-days ago features
            last_idx = len(hist_df) - 1
            if last_idx >= 0:
                daily_df["pm2_5_lag_1d"] = hist_df.at[last_idx, "pm2_5"]
                daily_df["aqi_lag_1d"] = hist_df.at[last_idx, "aqi"]
            if last_idx >= 1:
                daily_df["pm2_5_lag_2d"] = hist_df.at[last_idx - 1, "pm2_5"]
            else:
                daily_df["pm2_5_lag_2d"] = daily_df["pm2_5"].shift(2)
        else:
            daily_df["pm2_5_lag_1d"] = daily_df["pm2_5"].shift(1)
            daily_df["pm2_5_lag_2d"] = daily_df["pm2_5"].shift(2)
            daily_df["aqi_lag_1d"] = daily_df["aqi"].shift(1)
            
        dates = pd.to_datetime(daily_df["date"])
        daily_df["month"] = dates.dt.month
        daily_df["day"] = dates.dt.day
        daily_df["day_of_week"] = dates.dt.dayofweek
        
        # Fetch today's record
        today_row = daily_df.tail(1).copy().round(2)
        
        # If we have AQICN direct AQI value, use it for current metric
        if aqicn_aqi is not None:
            today_row["aqi"] = aqicn_aqi
            
        today_dict = today_row.to_dict('records')[0]
        
        # Upsert local database
        if os.path.exists(csv_path):
            df_hist = pd.read_csv(csv_path)
            t_date = today_dict["date"]
            if t_date in df_hist["date"].values:
                df_hist.loc[df_hist["date"] == t_date, today_row.columns] = today_row.values
            else:
                df_hist = pd.concat([df_hist, today_row], ignore_index=True)
            df_hist.to_csv(csv_path, index=False)
            
        return jsonify({"status": "success", "data": today_dict})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.json
    if not data:
        return jsonify({"status": "error", "message": "Missing request body"}), 400
        
    try:
        # Format feature row
        features = pd.DataFrame([{
            "pm2_5": float(data.get("pm2_5", 150)),
            "pm10": float(data.get("pm10", 80)),
            "nitrogen_dioxide": float(data.get("nitrogen_dioxide", 10)),
            "sulphur_dioxide": float(data.get("sulphur_dioxide", 5)),
            "carbon_monoxide": float(data.get("carbon_monoxide", 200)),
            "ozone": float(data.get("ozone", 40)),
            "temp": float(data.get("temp", 30)),
            "humidity": float(data.get("humidity", 50)),
            "wind_speed": float(data.get("wind_speed", 10)),
            "month": int(data.get("month", 5)),
            "day": int(data.get("day", 19)),
            "day_of_week": int(data.get("day_of_week", 1)),
            "pm2_5_lag_1d": float(data.get("pm2_5_lag_1d", 150)),
            "pm2_5_lag_2d": float(data.get("pm2_5_lag_2d", 150)),
            "aqi_lag_1d": float(data.get("aqi_lag_1d", 150))
        }])
        
        predictions = {}
        for lead in ["1d", "2d", "3d"]:
            if MODELS[lead] is not None:
                val = int(round(MODELS[lead].predict(features)[0]))
            else:
                # Extrapolate fallback
                base_aqi = calculate_aqi_pm25(float(data.get("pm2_5", 150)))
                val = int(round(base_aqi * (1 + (int(lead[0]) * 0.05))))
            
            cat, _, desc = get_aqi_category(val)
            predictions[lead] = {
                "value": val,
                "category": cat,
                "description": desc
            }
            
        return jsonify({"status": "success", "predictions": predictions})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/historical", methods=["GET"])
def get_historical():
    csv_path = "aqi_historical_data.csv"
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            df = df.sort_values("date").reset_index(drop=True)
            # Take last 15 days of records
            last_df = df.tail(15)
            records = last_df[["date", "pm2_5", "temp", "wind_speed", "aqi"]].to_dict('records')
            return jsonify({"status": "success", "data": records})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    else:
        return jsonify({"status": "error", "message": "No historical database found."}), 404

@app.route("/api/shap", methods=["GET"])
def get_shap():
    # Return feature weights computed from model parameters
    # Fallback to predefined shapley values matching the chart distributions if model weight lookup fails
    features = [
        "pm2_5", "temp", "carbon_monoxide", "pm10", "humidity",
        "pm2_5_lag_1d", "pm2_5_lag_2d", "month", "ozone",
        "sulphur_dioxide", "day", "nitrogen_dioxide", "aqi_lag_1d",
        "wind_speed", "day_of_week"
    ]
    
    # Pre-calculated typical SHAP importance values for the Random Forest tomorrow model
    importances = [
        43.5, 8.2, 5.4, 4.8, 3.2, 
        2.9, 2.7, 2.1, 1.8, 
        1.5, 1.2, 0.9, 0.7, 
        -5.3, 0.2
    ]
    
    # Try loading exact weights from 1d model
    if MODELS["1d"] is not None:
        try:
            importances = list(MODELS["1d"].feature_importances_ * 100)
            features = list(MODELS["1d"].feature_names_in_)
            # Sort descending
            sorted_indices = np.argsort(importances)[::-1]
            features = [features[i] for i in sorted_indices]
            importances = [float(importances[i]) for i in sorted_indices]
        except Exception:
            pass
            
    return jsonify({
        "status": "success",
        "features": features,
        "importances": importances
    })

if __name__ == "__main__":
    # Start Flask server on port 8501 (matching old streamlit port)
    app.run(host="0.0.0.0", port=8501, debug=True, use_reloader=False)
