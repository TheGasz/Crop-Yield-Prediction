from flask import Flask, request, jsonify
from flask_cors import CORS  
import numpy as np
from tensorflow.keras.models import load_model 
import joblib

app = Flask(__name__)
CORS(app) 

# Load model dan scaler
# Pastikan file-file ini ada di folder yang sama dengan main.py
try:
    # Coba load format .keras dulu (recommended)
    try:
        model = load_model("crop_yield_model.keras")
        print("✅ Loaded model from crop_yield_model.keras")
    except:
        # Fallback ke .h5 format
        model = load_model("model.h5")
        print("✅ Loaded model from model.h5")
    
    scaler = joblib.load("x_scaler.pkl")
    print("✅ Loaded scaler from x_scaler.pkl")
    
except Exception as e:
    print(f"❌ Error loading model or scaler: {e}")
    model = None
    scaler = None

@app.route('/')
def index():
    return "API Model Prediksi Hasil Pertanian Siap!"

# Ganti nama endpoint agar lebih jelas ini adalah API
@app.route('/api/predict', methods=['POST'])
def predict():
    if not model or not scaler:
        return jsonify({"error": "Model or scaler not loaded"}), 500

    try:
        data = request.get_json()

        # Pastikan semua fitur yang dibutuhkan ada di data JSON
        required_features = [
            'Soil_Quality','Seed_Variety','Fertilizer_Ammount_kg_per_hectare','Rainfall_mm',
            'Irrigation_Schedule','Yield_kg_per_hectare'
        ]
        
        if not all(feature in data for feature in required_features):
            return jsonify({"error": "Missing one or more features in the request"}), 400

        # Urutan fitur HARUS SAMA PERSIS dengan saat training
        features = [
            data['Soil_Quality'],
            data['Seed_Variety'],
            data['Fertilizer_Ammount_kg_per_hectare'],
            data['Rainfall_mm'],
            data['Irrigation_Schedule'],
            data['Yield_kg_per_hectare']
        ]

        # Preprocessing: scaling
        input_scaled = scaler.transform([features])

        # Prediksi
        prediction = model.predict(input_scaled)[0][0]
        return jsonify({"predicted_yield_kg_per_hectare": float(prediction)})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Jalankan di port 5000 agar tidak bentrok dengan React
    app.run(port=5000, debug=True)


# PEMBATAS, INI BEDA

# from flask import Flask, request, jsonify
# import numpy as np
# from tensorflow.keras.models import load_model
# import joblib

# app = Flask(__name__)

# # Load model dan scaler buatan sendiri
# model = load_model("yield_model.h5")
# scaler = joblib.load("x_scaler.pkl")

# @app.route('/')
# def index():
#     return "API Model Prediksi Hasil Pertanian Siap!"

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.get_json()

#         # Sesuaikan dengan fitur model kamu
#         features = [
#             data['Temperature_C'],SS
#             data['Rainfall_mm'],
#             data['Humidity_percent'],
#             data['Fertilizer_kg_per_hectare'],
#             data['Pesticide_kg_per_hectare'],
#             data['Soil_quality_index']
#         ]

#         # Preprocessing: scaling
#         input_scaled = scaler.transform([features])

#         # Prediksi
#         prediction = model.predict(input_scaled)[0][0]
#         return jsonify({"predicted_yield_kg_per_hectare": float(prediction)})
    
#     except Exception as e:
#         return jsonify({"error": str(e)})

# if __name__ == '__main__':
#     app.run(debug=True)
