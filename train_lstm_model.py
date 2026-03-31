"""
LSTM Training Script (Using Delhi AQI Dataset)
"""

import os
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping

# ---------------- CONFIG ----------------

DATA_PATH = "data/final_dataset.csv"
MODEL_SAVE_PATH = "models/aqi_lstm_model.keras"
SCALER_SAVE_PATH = "models/aqi_scaler.pkl"

WINDOW_SIZE = 7
EPOCHS = 50
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.1

# ---------------- FUNCTIONS ----------------

def handle_missing_values(df):
    return df.interpolate(method='linear')

def normalize_data(df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)
    return scaled, scaler

def create_lstm_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

def build_model(input_shape):
    model = Sequential()

    # 1. First Layer: Bidirectional LSTM with 100 units to capture patterns both ways
    model.add(Bidirectional(LSTM(100, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(0.2))

    # 2. Second Layer: Stacked LSTM for deeper feature extraction
    model.add(LSTM(100))
    model.add(Dropout(0.2))

    # 3. Output Layer: Single value prediction (Tomorrow's AQI)
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# ---------------- MAIN ----------------

def main():

    os.makedirs("models", exist_ok=True)

    # LOAD REAL DATA ONLY
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    print(f"Loading dataset from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    print("Columns:", df.columns)

    # Fix AQI column
    if 'AQI' not in df.columns:
        if 'aqi' in df.columns:
            df.rename(columns={'aqi': 'AQI'}, inplace=True)
        else:
            raise ValueError("AQI column not found")

    # Sort by date if exists
    if 'date' in df.columns:
        df = df.sort_values('date')

    # Select AQI only
    df = df[['AQI']]

    # Handle missing values
    print("Cleaning data...")
    df = handle_missing_values(df)

    # Normalize
    print("Scaling data...")
    scaled_data, scaler = normalize_data(df)

    # Save scaler
    joblib.dump(scaler, SCALER_SAVE_PATH)
    print(f"Scaler saved -> {SCALER_SAVE_PATH}")

    # Create sequences
    print("Creating sequences...")
    X, y = create_lstm_sequences(scaled_data, WINDOW_SIZE)

    print(f"Shape -> X: {X.shape}, y: {y.shape}")

    # Build model
    model = build_model((X.shape[1], X.shape[2]))
    model.summary()

    # Train
    print("Training model...")

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    model.fit(
        X, y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=[early_stop],
        verbose=1
    )

    # Save model
    model.save(MODEL_SAVE_PATH)
    print(f"\nModel saved -> {MODEL_SAVE_PATH}")
    print("Training completed!")

# ---------------- RUN ----------------

if __name__ == "__main__":
    main()