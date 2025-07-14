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
#             data['Temperature_C'],
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
