"""
Data Preprocessing Service for Deep Learning (LSTM)
Handles data cleaning, normalization, and sequence generation specifically tailored for Air Quality metrics.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def handle_missing_values(df: pd.DataFrame, method: str = 'interpolate') -> pd.DataFrame:
    """
    Cleans datasets by processing missing and invalid values.
    Since environmental sensor data is continuous, interpolation is heavily favored.
    
    Args:
        df: Input pandas DataFrame
        method: Method to handle NaNs ('interpolate', 'forward_fill', 'backward_fill', 'drop')
        
    Returns:
        Cleaned pandas DataFrame
    """
    clean_df = df.copy()
    
    if method == 'interpolate':
        # Linear interpolation fills missing time-gaps intelligently
        clean_df = clean_df.interpolate(method='linear', limit_direction='both')
    elif method == 'forward_fill':
        clean_df = clean_df.ffill()
    elif method == 'backward_fill':
        clean_df = clean_df.bfill()
    elif method == 'drop':
        clean_df = clean_df.dropna()
    else:
        raise ValueError(f"Unknown handling method: '{method}'")
        
    # Safety catch for persistent NaNs
    clean_df = clean_df.fillna(0)
    
    return clean_df


def normalize_data(df: pd.DataFrame, feature_columns: list, scaler: MinMaxScaler = None):
    """
    Normalizes specific columns to a tight bounded range (0 to 1) via MinMaxScaler 
    to assist the LSTM gradient descent from vanishing or exploding gradients.
    
    Args:
        df: Input pandas DataFrame
        feature_columns: List of columns to scale (e.g., ['pm2_5', 'pm10', 'no2'])
        scaler: Pre-instantiated MinMaxScaler (Pass this during test/inference phases)
        
    Returns:
        Tuple: (Scaled DataFrame, Fitted MinMaxScaler)
    """
    scaled_df = df.copy()
    
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        # Fit logic: Training Phase
        scaled_features = scaler.fit_transform(scaled_df[feature_columns])
    else:
        # Transform strictly: Testing/Inference Phase
        scaled_features = scaler.transform(scaled_df[feature_columns])
        
    scaled_df[feature_columns] = scaled_features
    
    return scaled_df, scaler


def create_lstm_sequences(data: np.ndarray, window_size: int, target_col_idx: int = None):
    """
    Restructures 2D continuous time-series arrays into 3D sliding-window tensor chunks 
    expected by Keras / PyTorch LSTM layers directly.
    
    Layout transformation:
        2D (samples, features) -> 3D (samples, window_size, features).
    
    Args:
        data: 2D numpy array containing the historically ordered records
        window_size: Look-back size (How many previous steps define 1 sequence matrix)
        target_col_idx: Integer. If provided, returns X (features) and y (target label).
                        If None, simply creates X chunks (Used for predictions).
                        
    Returns:
        If target_col_idx provided: (np.array of X sequences, np.array of y targets)
        If target_col_idx is None: np.array of X sequences
    """
    X, y = [], []
    
    dataset_length = len(data)
    if dataset_length <= window_size:
        raise ValueError(f"Dataset too small. Must have more records ({dataset_length}) than window_size ({window_size}).")
        
    for i in range(dataset_length - window_size):
        # Slice sliding window for the sequence X
        seq_x = data[i : i + window_size]
        X.append(seq_x)
        
        # Determine the target value logically immediately following the final window step
        if target_col_idx is not None:
            seq_y = data[i + window_size, target_col_idx]
            y.append(seq_y)
            
    if target_col_idx is not None:
        return np.array(X), np.array(y)
    
    return np.array(X)
