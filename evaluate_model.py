import os
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = "data/final_dataset.csv"
MODEL_PATH = "models/aqi_lstm_model.keras"
SCALER_PATH = "models/aqi_scaler.pkl"
WINDOW_SIZE = 7

def evaluate():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH) or not os.path.exists(DATA_PATH):
        print("Missing required files for evaluation.")
        return

    # Load artifacts
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    df = pd.read_csv(DATA_PATH)

    # Preprocess
    df = df[['AQI']].interpolate(method='linear')
    scaled_data = scaler.transform(df)

    X, y = [], []
    for i in range(len(scaled_data) - WINDOW_SIZE):
        X.append(scaled_data[i:i+WINDOW_SIZE])
        y.append(scaled_data[i+WINDOW_SIZE])
    
    X, y = np.array(X), np.array(y)

    # Predict
    predictions_scaled = model.predict(X)
    
    # Scale back
    y_true = scaler.inverse_transform(y.reshape(-1, 1))
    y_pred = scaler.inverse_transform(predictions_scaled)

    # Metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score: {r2:.4f}")

    # Accuracy percentage (Estimated)
    # Using 1 - (MAE / Mean_AQI) as a rough accuracy metric
    mean_aqi = np.mean(y_true)
    accuracy = (1 - (mae / mean_aqi)) * 100
    print(f"Estimated Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    evaluate()
