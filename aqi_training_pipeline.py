import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Try importing SHAP for model explanation
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

def load_data():
    csv_path = "aqi_historical_data.csv"
    
    # Try fetching from Hopsworks first
    try:
        import hopsworks
        print("Attempting to connect to Hopsworks to fetch training data...")
        project = hopsworks.login()
        fs = project.get_feature_store()
        aqi_fg = fs.get_feature_group(name="aqi_karachi_fg", version=1)
        df = aqi_fg.read()
        print(f"Successfully loaded {len(df)} rows from Hopsworks Feature Store.")
        # Sort by date
        df = df.sort_values("date").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"Could not load from Hopsworks ({e}). Trying local CSV fallback...")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            print(f"Loaded {len(df)} rows from local {csv_path}.")
            df = df.sort_values("date").reset_index(drop=True)
            return df
        else:
            raise FileNotFoundError(f"Neither Hopsworks connection nor local {csv_path} was found. Please run the backfill script first.")

def prepare_targets_and_features(df):
    print("Preparing training features and targets (1-day, 2-day, 3-day lead AQI)...")
    
    # Targets: AQI in next 1, 2, and 3 days
    df["target_aqi_1d"] = df["aqi"].shift(-1)
    df["target_aqi_2d"] = df["aqi"].shift(-2)
    # Open-Meteo data is clean, but let's drop rows without targets for training
    df_clean = df.dropna(subset=["target_aqi_1d", "target_aqi_2d"]).copy()
    
    # Wait, 3-day forecast needs shift(-3)
    df["target_aqi_3d"] = df["aqi"].shift(-3)
    df_clean = df.dropna(subset=["target_aqi_1d", "target_aqi_2d", "target_aqi_3d"]).copy()
    
    # Define features
    feature_cols = [
        "pm2_5", "pm10", "nitrogen_dioxide", "sulphur_dioxide", "carbon_monoxide", "ozone",
        "temp", "humidity", "wind_speed", "month", "day", "day_of_week",
        "pm2_5_lag_1d", "pm2_5_lag_2d", "aqi_lag_1d"
    ]
    
    # Ensure all feature columns exist and don't contain NaNs
    df_clean = df_clean.dropna(subset=feature_cols)
    
    X = df_clean[feature_cols]
    y_1d = df_clean["target_aqi_1d"]
    y_2d = df_clean["target_aqi_2d"]
    y_3d = df_clean["target_aqi_3d"]
    
    return X, y_1d, y_2d, y_3d, feature_cols, df_clean

def train_and_evaluate(X, y, target_name):
    # Time-series split: use last 30 days as validation
    split_idx = len(X) - 30
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"\n--- Training Model for {target_name} ---")
    print(f"Train set: {X_train.shape[0]} rows, Test set: {X_test.shape[0]} rows")
    
    # Initialize Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    print(f"Evaluation Metrics for {target_name}:")
    print(f"  Mean Absolute Error (MAE): {mae:.2f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"  R2 Score: {r2:.3f}")
    
    # Retrain on full dataset before saving
    final_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    final_model.fit(X, y)
    
    return final_model, r2

def save_model_local(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Saved model to local file {filename}")

def register_model_hopsworks(model_path, model_name, r2_score_val):
    try:
        import hopsworks
        print(f"Registering {model_name} in Hopsworks Model Registry...")
        project = hopsworks.login()
        mr = project.get_model_registry()
        
        # Create model metadata
        mr_model = mr.python.create_model(
            name=model_name,
            metrics={"r2": r2_score_val},
            description=f"Random Forest model for forecasting AQI {model_name.split('_')[-1]} in Karachi."
        )
        mr_model.save(model_path)
        print(f"Successfully registered {model_name} in Model Registry!")
    except Exception as e:
        print(f"Could not register in Model Registry: {e}")

def generate_shap_plot(model, X, feature_cols):
    if not SHAP_AVAILABLE:
        print("SHAP library is not installed. Skipping SHAP explanation plot.")
        return
        
    print("\nGenerating SHAP feature explanations...")
    # Use TreeExplainer for Random Forest
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # Plot summary
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, feature_names=feature_cols, show=False)
    plt.title("Feature Importance Breakdown (SHAP Values) - 1-Day Forecast", fontsize=14, pad=15)
    plt.tight_layout()
    plot_path = "shap_summary_1d.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved SHAP plot to {plot_path}")

def main():
    # Load dataset
    df = load_data()
    
    # Prepare features and targets
    X, y_1d, y_2d, y_3d, feature_cols, df_clean = prepare_targets_and_features(df)
    
    # Train models
    model_1d, r2_1d = train_and_evaluate(X, y_1d, "1-Day Lead AQI (Tomorrow)")
    model_2d, r2_2d = train_and_evaluate(X, y_2d, "2-Day Lead AQI (Day After)")
    model_3d, r2_3d = train_and_evaluate(X, y_3d, "3-Day Lead AQI (3 Days)")
    
    # Save models locally
    save_model_local(model_1d, "aqi_model_1d.pkl")
    save_model_local(model_2d, "aqi_model_2d.pkl")
    save_model_local(model_3d, "aqi_model_3d.pkl")
    
    # Generate explanations
    generate_shap_plot(model_1d, X, feature_cols)
    
    # Ask to register models in Hopsworks
    ans = input("\nDo you want to register these models to Hopsworks Model Registry? (y/n): ")
    if ans.lower() == 'y':
        register_model_hopsworks("aqi_model_1d.pkl", "aqi_model_1d", r2_1d)
        register_model_hopsworks("aqi_model_2d.pkl", "aqi_model_2d", r2_2d)
        register_model_hopsworks("aqi_model_3d.pkl", "aqi_model_3d", r2_3d)

if __name__ == "__main__":
    main()
