import os
import time
import requests
from services.inference_service import predict_aqi

def get_aqi_category(aqi_index):
    """Map OpenWeatherMap AQI (1-5) to human-readable categories."""
    categories = {
        1: {"status": "Good", "class": "good", "rec": "Air quality is satisfactory. Enjoy outdoor activities!"},
        2: {"status": "Fair", "class": "good", "rec": "Air quality is acceptable. Unusually sensitive people should consider reducing prolonged outdoor exertion."},
        3: {"status": "Moderate", "class": "warning", "rec": "Members of sensitive groups may experience health effects. The general public is not likely to be affected."},
        4: {"status": "Poor", "class": "danger", "rec": "Everyone may begin to experience health effects. Wear a mask outdoors and reduce physical exertion."},
        5: {"status": "Very Poor", "class": "danger", "rec": "EMERGENCY: Health warnings of emergency conditions. Stay indoors, keep windows closed, and use an air purifier."}
    }
    return categories.get(aqi_index, {"status": "Unknown", "class": "neutral", "rec": "No recommendations available."})

def get_predicted_aqi_category(predicted_value):
    """Map predicted float AQI dynamically to requested Alert Categories."""
    if predicted_value is None:
        return {"status": "Unknown", "class": "neutral", "rec": ""}
        
    rec_good = "Tomorrow's forecast looks clear. Perfect for outdoor activities!"
    rec_mod = "Forecast shows elevated pollutants tomorrow. Sensitive groups should prepare."
    rec_unhealthy = "Forecast predicts unhealthy air tomorrow. Plan to wear an N95 mask outdoors."
    rec_haz = "HAZARDOUS forecast tomorrow. Prepare to stay indoors and keep windows sealed."
    
    # Handling both OWM scale (1-5) and Standard EPA scale (0-300) dynamically
    if predicted_value <= 5.0:
        if predicted_value <= 1.5: return {"status": "Good", "class": "good", "rec": rec_good}
        elif predicted_value <= 2.5: return {"status": "Moderate", "class": "warning", "rec": rec_mod}
        elif predicted_value <= 4.0: return {"status": "Unhealthy", "class": "danger", "rec": rec_unhealthy}
        else: return {"status": "Hazardous", "class": "danger", "rec": rec_haz}
    else:
        if predicted_value <= 50: return {"status": "Good", "class": "good", "rec": rec_good}
        elif predicted_value <= 100: return {"status": "Moderate", "class": "warning", "rec": rec_mod}
        elif predicted_value <= 200: return {"status": "Unhealthy", "class": "danger", "rec": rec_unhealthy}
        else: return {"status": "Hazardous", "class": "danger", "rec": rec_haz}

def convert_aqi_scale(owm_aqi):
    """Convert OWM AQI (1-5) to Standard EPA Approximate (0-500)."""
    mapping = {1: 25, 2: 75, 3: 125, 4: 175, 5: 250}
    return mapping.get(owm_aqi, 100)

def get_city_air_quality(city_name):
    """
    Fetch coordinates and then current air quality data for a given city.
    Returns:
        dict: {'success': True/False, 'data': {...}, 'error': 'error message'}
    """
    api_key = os.environ.get('OPENWEATHER_API_KEY')
    if not api_key or api_key == 'your_api_key_here':
        return {'success': False, 'error': 'API key is missing or invalid. Please check .env file.'}
    
    if not city_name or not city_name.strip():
         return {'success': False, 'error': 'City name is required.'}

    # Step 1: Geocoding API - Get latitude and longitude
    geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city_name}&limit=1&appid={api_key}"
    try:
        # Use timeout to prevent hanging requests
        geo_response = requests.get(geo_url, timeout=10)
        
        # OpenWeather returns 401 for invalid API keys on this endpoint too
        if geo_response.status_code == 401:
            return {'success': False, 'error': 'Invalid API Key. Please verify OPENWEATHER_API_KEY in .env.'}
            
        geo_response.raise_for_status()
        geo_data = geo_response.json()
        
        if not geo_data or len(geo_data) == 0:
            return {'success': False, 'error': f'City "{city_name}" not found.'}
            
        lat = geo_data[0]['lat']
        lon = geo_data[0]['lon']
        actual_city_name = geo_data[0].get('name', city_name.title())
        country = geo_data[0].get('country', '')
        
    except requests.exceptions.RequestException as e:
        return {'success': False, 'error': f'Geocoding API network error: {str(e)}'}

    # Step 2: Air Pollution API
    aq_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
    try:
        aq_response = requests.get(aq_url, timeout=10)
        aq_response.raise_for_status()
        aq_data = aq_response.json()
        
        if 'list' not in aq_data or len(aq_data['list']) == 0:
            return {'success': False, 'error': 'No air quality data available for this location.'}
            
        pollution_info = aq_data['list'][0]
        aqi_index = pollution_info['main']['aqi']
        components = pollution_info['components']
        
        category = get_aqi_category(aqi_index)
        
        # ML Inference: Attempt to silently fetch historical data and generate a prediction
        predicted_aqi = None
        historical_aqi = []
        try:
            end_time = int(time.time())
            start_time = end_time - (7 * 24 * 60 * 60) # 7 Days precisely
            hist_url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={lat}&lon={lon}&start={start_time}&end={end_time}&appid={api_key}"
            hist_response = requests.get(hist_url, timeout=5)
            
            if hist_response.status_code == 200:
                hist_data = hist_response.json()
                if 'list' in hist_data and len(hist_data['list']) >= 7:
                    # History payload is dense hourly data. Extract daily chunks
                    daily_samples = [item['main']['aqi'] for item in hist_data['list']]
                    step = max(1, len(daily_samples) // 7)
                    sequence_owm = daily_samples[::step][-7:]
                    
                    # Pad array gracefully backward if network payload was shorter than 7 days
                    while len(sequence_owm) < 7:
                        sequence_owm.insert(0, sequence_owm[0])
                        
                    # Exactly 7 required for LSTM Tensor
                    sequence_owm = sequence_owm[:7]
                    
                    # 🔥 CRITICAL: Convert OWM Scale (1-5) to Model Scale (0-500)
                    sequence_standard = [convert_aqi_scale(val) for val in sequence_owm]
                    
                    historical_aqi = sequence_standard
                    predicted_aqi = predict_aqi(sequence_standard)
                    
        except Exception as e:
            # Shield live traffic returning 500s from non-critical ML model loading errors
            pass
            
        # Safety evaluation for predicted UI classes
        predicted_alert = get_predicted_aqi_category(predicted_aqi)
        
        # Package the result cleanly
        result_data = {
            'location': {
                'city': actual_city_name,
                'country': country,
                'lat': lat,
                'lon': lon
            },
            'aqi': aqi_index,
            'historical_aqi': historical_aqi,
            'recommendation': category['rec'],
            'predicted_aqi': predicted_aqi,
            'predicted_status': predicted_alert['status'],
            'predicted_class': predicted_alert['class'],
            'predicted_recommendation': predicted_alert.get('rec', ''),
            'status': category['status'],
            'css_class': category['class'],
            'pollutants': {
                'pm2_5': components.get('pm2_5', 0),
                'pm10': components.get('pm10', 0),
                'no2': components.get('no2', 0),
                'so2': components.get('so2', 0),
                'co': components.get('co', 0),
                'o3': components.get('o3', 0)
            }
        }
        return {'success': True, 'data': result_data}
        
    except requests.exceptions.RequestException as e:
        return {'success': False, 'error': f'Air Pollution API network error: {str(e)}'}
