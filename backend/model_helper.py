"""
Model Loading Helper - Crop Yield Prediction
============================================

Cara mudah untuk load dan menggunakan model:

from model_helper import load_model_easy, predict_easy

# Load model
model, scaler = load_model_easy()

# Predict
result = predict_easy(model, scaler, 
                     temperature=25, rainfall=150, humidity=70,
                     fertilizer=50, pesticide=5, soil_quality=80)
print(f"Predicted yield: {result} kg/hectare")
"""

import joblib
from tensorflow.keras.models import load_model
import numpy as np

def load_model_easy(model_path='crop_yield_model.keras', scaler_path='x_scaler.pkl'):
    """
    Load model dan scaler dengan mudah
    
    Returns:
        tuple: (model, scaler) or (None, None) if failed
    """
    try:
        # Try .keras format first
        try:
            model = load_model(model_path)
            print(f"✅ Model loaded from {model_path}")
        except:
            # Fallback to .h5
            model = load_model('model.h5')
            print("✅ Model loaded from model.h5")
        
        scaler = joblib.load(scaler_path)
        print(f"✅ Scaler loaded from {scaler_path}")
        
        return model, scaler
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None

def predict_easy(model, scaler, temperature, rainfall, humidity, 
                fertilizer, pesticide, soil_quality):
    """
    Prediksi yang mudah dengan parameter terpisah
    
    Args:
        model: loaded model
        scaler: loaded scaler
        temperature: Temperature in Celsius
        rainfall: Rainfall in mm
        humidity: Humidity in percentage
        fertilizer: Fertilizer in kg per hectare
        pesticide: Pesticide in kg per hectare
        soil_quality: Soil quality index
    
    Returns:
        float: predicted yield in kg/hectare
    """
    if not model or not scaler:
        print("❌ Model or scaler not loaded!")
        return None
    
    try:
        features = [temperature, rainfall, humidity, fertilizer, pesticide, soil_quality]
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0][0]
        return float(prediction)
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return None

# Demo usage
if __name__ == "__main__":
    print("🧪 Testing Model Helper")
    print("="*30)
    
    # Load model
    model, scaler = load_model_easy()
    
    if model and scaler:
        # Test prediction
        result = predict_easy(
            model, scaler,
            temperature=25.0,
            rainfall=150.0, 
            humidity=70.0,
            fertilizer=50.0,
            pesticide=5.0,
            soil_quality=80.0
        )
        
        if result:
            print(f"\n🌾 Predicted Yield: {result:.2f} kg/hectare")
        else:
            print("❌ Prediction failed!")
    else:
        print("❌ Model loading failed!")
