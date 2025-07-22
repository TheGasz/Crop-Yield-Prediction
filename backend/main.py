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
            'Soil_Quality','Seed_Variety','Sunny_Days','Fertilizer_Ammount_kg_per_hectare','Rainfall_mm',
            'Irrigation_Schedule'
        ]
        
        if not all(feature in data for feature in required_features):
            return jsonify({"error": "Missing one or more features in the request"}), 400

        # Ekstrak nilai input dasar
        soil_quality = data['Soil_Quality']
        seed_variety = data['Seed_Variety']
        sunny_days = data['Sunny_Days']
        fertilizer_amount = data['Fertilizer_Ammount_kg_per_hectare']
        rainfall = data['Rainfall_mm']
        irrigation = data['Irrigation_Schedule']
        
        # Feature Engineering - sama seperti di training
        log_fertilizer = np.log1p(fertilizer_amount)
        log_rainfall = np.log1p(rainfall)
        log_irrigation = np.log1p(irrigation)
        
        total_water = log_rainfall + log_irrigation * 50
        fertilizer_per_water = log_fertilizer / (total_water + 1)
        seed_irrigation_interaction = seed_variety * log_irrigation
        fertilizer_per_irrigation = log_fertilizer / (irrigation + 1)
        
        # Urutan fitur HARUS SAMA PERSIS dengan saat training
        # Berdasarkan X_train = yield_dataset.drop('Yield_kg_per_hectare', axis=1)
        features = [
            soil_quality,                     # 0: Soil_Quality
            seed_variety,                     # 1: Seed_Variety
            fertilizer_amount,                # 2: Fertilizer_Amount_kg_per_hectare
            sunny_days,                       # 3: Sunny_Days
            rainfall,                         # 4: Rainfall_mm
            irrigation,                       # 5: Irrigation_Schedule
            log_fertilizer,                   # 6: log_Fertilizer_Amount_kg_per_hectare
            log_rainfall,                     # 7: log_Rainfall_mm
            log_irrigation,                   # 8: log_Irrigation_Schedule
            total_water,                      # 9: Total_Water
            fertilizer_per_water,             # 10: Fertilizer_per_Water
            seed_irrigation_interaction,      # 11: Seed_Irrigation_Interaction
            fertilizer_per_irrigation         # 12: Fertilizer_per_Irrigation
        ]

        # Preprocessing: scaling
        input_scaled = scaler.transform([features])

        # Prediksi
        prediction = model.predict(input_scaled)[0][0]
        return jsonify({"predicted_yield_kg_per_hectare": float(prediction)})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/test', methods=['GET'])
def test_prediction():
    """Test endpoint dengan data contoh"""
    if not model or not scaler:
        return jsonify({"error": "Model or scaler not loaded"}), 500
    
    try:
        # Data contoh yang realistis
        test_data = {
            'Soil_Quality': 80,
            'Seed_Variety': 1,
            'Fertilizer_Ammount_kg_per_hectare': 50,
            'Sunny_Days': 25,
            'Rainfall_mm': 150,
            'Irrigation_Schedule': 3
        }
        
        # Ekstrak nilai input dasar
        soil_quality = test_data['Soil_Quality']
        seed_variety = test_data['Seed_Variety']
        sunny_days = test_data['Sunny_Days']
        fertilizer_amount = test_data['Fertilizer_Ammount_kg_per_hectare']
        rainfall = test_data['Rainfall_mm']
        irrigation = test_data['Irrigation_Schedule']
        
        # Feature Engineering
        log_fertilizer = np.log1p(fertilizer_amount)
        log_rainfall = np.log1p(rainfall)
        log_irrigation = np.log1p(irrigation)
        
        total_water = log_rainfall + log_irrigation * 50
        fertilizer_per_water = log_fertilizer / (total_water + 1)
        seed_irrigation_interaction = seed_variety * log_irrigation
        fertilizer_per_irrigation = log_fertilizer / (irrigation + 1)
        
        # Urutan fitur sesuai training
        features = [
            soil_quality,                     # 0: Soil_Quality
            seed_variety,                     # 1: Seed_Variety
            fertilizer_amount,                # 2: Fertilizer_Amount_kg_per_hectare
            sunny_days,                       # 3: Sunny_Days
            rainfall,                         # 4: Rainfall_mm
            irrigation,                       # 5: Irrigation_Schedule
            log_fertilizer,                   # 6: log_Fertilizer_Amount_kg_per_hectare
            log_rainfall,                     # 7: log_Rainfall_mm
            log_irrigation,                   # 8: log_Irrigation_Schedule
            total_water,                      # 9: Total_Water
            fertilizer_per_water,             # 10: Fertilizer_per_Water
            seed_irrigation_interaction,      # 11: Seed_Irrigation_Interaction
            fertilizer_per_irrigation         # 12: Fertilizer_per_Irrigation
        ]
        
        # Preprocessing: scaling
        input_scaled = scaler.transform([features])
        
        # Prediksi
        prediction = model.predict(input_scaled)[0][0]
        
        return jsonify({
            "test_input": test_data,
            "processed_features": features,
            "predicted_yield_kg_per_hectare": float(prediction),
            "message": "Test prediction berhasil"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
    """Generate detailed recommendations based on input data"""
    try:
        data = request.get_json()
        
        recommendations = []
        
        # Analisis dan rekomendasi pupuk
        fertilizer = data.get('Fertilizer_Ammount_kg_per_hectare', 0)
        if fertilizer < 30:
            recommendations.append({
                "category": "Pupuk",
                "priority": "Tinggi",
                "issue": "Pupuk kurang",
                "recommendation": f"Tingkatkan pupuk dari {fertilizer} kg/hektar menjadi 40-60 kg/hektar",
                "expected_impact": "Peningkatan hasil 15-25%",
                "implementation": "Aplikasi pupuk NPK secara bertahap sesuai fase pertumbuhan tanaman"
            })
        elif fertilizer > 80:
            recommendations.append({
                "category": "Pupuk", 
                "priority": "Sedang",
                "issue": "Pupuk berlebihan",
                "recommendation": f"Kurangi pupuk dari {fertilizer} kg/hektar menjadi 60-70 kg/hektar",
                "expected_impact": "Menghemat biaya 20-30% tanpa mengurangi hasil",
                "implementation": "Hitung ulang kebutuhan pupuk berdasarkan uji tanah"
            })
        
        # Analisis irigasi
        irrigation = data.get('Irrigation_Schedule', 0)
        if irrigation < 2:
            recommendations.append({
                "category": "Irigasi",
                "priority": "Tinggi", 
                "issue": "Irigasi kurang",
                "recommendation": f"Tingkatkan frekuensi irigasi dari {irrigation} menjadi 3-4 kali per minggu",
                "expected_impact": "Peningkatan hasil 10-20%",
                "implementation": "Buat jadwal irigasi rutin, monitor kelembaban tanah"
            })
        elif irrigation > 7:
            recommendations.append({
                "category": "Irigasi",
                "priority": "Sedang",
                "issue": "Irigasi berlebihan",
                "recommendation": f"Kurangi frekuensi irigasi dari {irrigation} menjadi 3-4 kali per minggu", 
                "expected_impact": "Mencegah akar busuk, menghemat air 30%",
                "implementation": "Gunakan sensor kelembaban tanah untuk irigasi presisi"
            })
        
        # Analisis kualitas tanah
        soil_quality = data.get('Soil_Quality', 0)
        if soil_quality < 60:
            recommendations.append({
                "category": "Tanah",
                "priority": "Tinggi",
                "issue": "Kualitas tanah rendah",
                "recommendation": f"Perbaiki kualitas tanah dari {soil_quality}/100",
                "expected_impact": "Peningkatan hasil jangka panjang 25-40%",
                "implementation": "Tambahkan kompos 2-3 ton/hektar, lakukan rotasi tanaman"
            })
        
        # Analisis curah hujan
        rainfall = data.get('Rainfall_mm', 0)
        if rainfall < 100:
            recommendations.append({
                "category": "Manajemen Air",
                "priority": "Sedang",
                "issue": "Curah hujan rendah",
                "recommendation": f"Kompensasi curah hujan rendah ({rainfall}mm)",
                "expected_impact": "Mencegah stress air pada tanaman",
                "implementation": "Tingkatkan irigasi, gunakan mulsa organik untuk konservasi air"
            })
        
        # Analisis hari cerah
        sunny_days = data.get('Sunny_Days', 0)
        if sunny_days < 20:
            recommendations.append({
                "category": "Manajemen Iklim",
                "priority": "Rendah",
                "issue": "Sinar matahari kurang",
                "recommendation": f"Optimalkan cahaya matahari ({sunny_days} hari cerah)",
                "expected_impact": "Peningkatan fotosintesis dan hasil",
                "implementation": "Pilih varietas toleran naungan, atur jarak tanam optimal"
            })
        
        return jsonify({
            "recommendations": recommendations,
            "summary": f"Ditemukan {len(recommendations)} area untuk peningkatan hasil panen"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat_response():
    """Simple chatbot responses for agricultural questions"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').lower()
        
        # Simple rule-based chatbot
        if any(word in user_message for word in ['pupuk', 'fertilizer', 'npk']):
            response = {
                "message": "🌱 Tips Pemupukan:\n\n" +
                          "• Gunakan pupuk NPK dengan rasio 15-15-15 untuk tanaman umum\n" +
                          "• Aplikasi pupuk dasar 60% saat tanam, 40% saat tanaman berumur 30 hari\n" +
                          "• Tambahkan pupuk organik 2-3 ton/hektar untuk kesehatan tanah jangka panjang\n" +
                          "• Lakukan uji tanah setiap musim untuk menentukan kebutuhan pupuk yang tepat",
                "type": "advice"
            }
        elif any(word in user_message for word in ['irigasi', 'air', 'siram']):
            response = {
                "message": "💧 Strategi Irigasi Optimal:\n\n" +
                          "• Irigasi pagi hari (06:00-08:00) atau sore hari (16:00-18:00)\n" +
                          "• Frekuensi 3-4 kali per minggu untuk hasil optimal\n" +
                          "• Gunakan sistem tetes untuk efisiensi air hingga 50%\n" +
                          "• Monitor kelembaban tanah dengan finger test atau sensor\n" +
                          "• Hindari irigasi berlebihan yang dapat menyebabkan akar busuk",
                "type": "advice"
            }
        elif any(word in user_message for word in ['tanah', 'soil', 'kompos']):
            response = {
                "message": "🌱 Manajemen Kualitas Tanah:\n\n" +
                          "• pH ideal tanah 6.0-7.0 untuk penyerapan nutrisi optimal\n" +
                          "• Tambahkan bahan organik 20-30% dari total volume tanah\n" +
                          "• Lakukan rotasi tanaman untuk mencegah deplesi nutrisi\n" +
                          "• Gunakan cover crop untuk melindungi struktur tanah\n" +
                          "• Hindari pengolahan tanah berlebihan saat basah",
                "type": "advice"
            }
        elif any(word in user_message for word in ['hama', 'penyakit', 'pest']):
            response = {
                "message": "🐛 Pengendalian Hama & Penyakit:\n\n" +
                          "• Terapkan sistem IPM (Integrated Pest Management)\n" +
                          "• Gunakan pestisida alami seperti neem oil atau ekstrak bawang putih\n" +
                          "• Tanam tanaman pengusir hama di sekitar lahan\n" +
                          "• Monitor rutin setiap 3-5 hari untuk deteksi dini\n" +
                          "• Jaga kebersihan lahan dari gulma dan sisa tanaman",
                "type": "advice"
            }
        elif any(word in user_message for word in ['varietas', 'benih', 'seed']):
            response = {
                "message": "🌾 Pemilihan Varietas Unggul:\n\n" +
                          "• Pilih varietas yang sesuai dengan iklim dan ketinggian lokasi\n" +
                          "• Gunakan benih bersertifikat untuk kualitas terjamin\n" +
                          "• Pertimbangkan varietas tahan hama dan penyakit lokal\n" +
                          "• Sesuaikan masa panen dengan rencana pemasaran\n" +
                          "• Lakukan uji coba skala kecil sebelum tanam massal",
                "type": "advice"
            }
        else:
            response = {
                "message": "🤖 Saya siap membantu Anda dengan:\n\n" +
                          "• Strategi pemupukan yang tepat\n" +
                          "• Teknik irigasi efisien\n" +
                          "• Manajemen kualitas tanah\n" +
                          "• Pengendalian hama dan penyakit\n" +
                          "• Pemilihan varietas unggul\n\n" +
                          "Silakan tanyakan topik yang Anda inginkan!",
                "type": "general"
            }
        
        return jsonify(response)
        
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
