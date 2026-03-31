import os
import joblib
import numpy as np
import logging

# Suppress annoying TF warnings during Flask boot
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow is not installed. LSTM Inference will be disabled.")

MODEL_PATH = 'models/aqi_lstm_model.keras'
SCALER_PATH = 'models/aqi_scaler.pkl'

# Lazy-loaded globals to prevent blocking app startup or crashing if missing
_model = None
_scaler = None

def _load_artifacts():
    """Internal helper to memory-load the ML artifacts safely exactly once."""
    global _model, _scaler
    
    if not TF_AVAILABLE:
        print("Model loading failed: TensorFlow is not installed.")
        return False
        
    try:
        # Use absolute paths to be safe relative to project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(project_root, 'models', 'aqi_lstm_model.keras')
        scaler_path = os.path.join(project_root, 'models', 'aqi_scaler.pkl')

        if _model is None and os.path.exists(model_path):
            _model = load_model(model_path)
            print("ML Model Loaded Successfully")
            
        if _scaler is None and os.path.exists(scaler_path):
            _scaler = joblib.load(scaler_path)
            print("ML Scaler Loaded Successfully")
            
        return _model is not None and _scaler is not None
    except Exception as e:
        print(f"Failed to load ML artifacts: {e}")
        return False

def predict_aqi(historical_sequence: list) -> float:
    """
    Takes exactly 7 previous days of AQI values, processes them, 
    and returns tomorrow's predicted AQI score using the LSTM model.
    
    Args:
        historical_sequence: A list/array of integers/floats (e.g. [45, 42, 50, 60, 55, 48, 42])
                             Must be the exact length the model was trained upon (window_size=7).
                             
    Returns:
        float: The predicted unscaled real-world AQI, or None if the model failed to process.
    """
    if not _load_artifacts():
        return None
        
    if len(historical_sequence) != 7:
        logging.error(f"Prediction failed. Expected sequence length 7, got {len(historical_sequence)}.")
        return None
        
    try:
        # Convert simple list to a 2D vertical array: [[45], [42], ...]
        seq_array = np.array(historical_sequence, dtype=np.float32).reshape(-1, 1)
        
        # 1. Prepare: Normalize incoming 7-day array utilizing the identical ratios tracked in training
        scaled_seq = _scaler.transform(seq_array)
        
        # 2. Reshape: Transform strictly from 2D -> 3D Tensor expected by Keras (1 sample, 7 timesteps, 1 feature)
        tensor_input = scaled_seq.reshape(1, 7, 1)
        
        # 3. Predict: Run the forward-pass
        scaled_prediction = _model.predict(tensor_input, verbose=0)
        
        # 4. Resolve: Inverse the bounding scaler on the specific raw 2D output
        real_value_prediction = _scaler.inverse_transform(scaled_prediction)
        
        # Extract the literal isolated float value mathematically
        raw_float_value = float(real_value_prediction[0][0])
        
        # Return a cleanly rounded precision score since AQI usually scales flatly
        return round(raw_float_value, 2)
        
    except Exception as e:
        logging.error(f"Error during LSTM prediction engine payload execution: {e}")
        return None
