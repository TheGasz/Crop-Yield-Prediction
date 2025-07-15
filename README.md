Proyek Prediksi Hasil Panen
Proyek ini adalah aplikasi web full-stack untuk memprediksi hasil panen pertanian berdasarkan beberapa fitur lingkungan dan pertanian.

Backend: API yang dibangun dengan Flask (Python) untuk melayani model prediksi Machine Learning (TensorFlow/Keras).

Frontend: Antarmuka pengguna interaktif yang dibangun dengan React.js.

Cara Menjalankan Proyek
Prasyarat
Python 3.8+ dan pip

Node.js dan npm

1. Setup Backend
# Masuk ke direktori backend
cd backend

# (Opsional tapi direkomendasikan) Buat dan aktifkan virtual environment
python -m venv venv
source venv/bin/activate  # Di Windows: venv\Scripts\activate

# Instal dependensi Python
# (Anda perlu membuat file requirements.txt terlebih dahulu)
pip install -r requirements.txt

# Jalankan server Flask
python main.py

Server backend akan berjalan di http://localhost:5000.

2. Setup Frontend
# Dari direktori root, masuk ke direktori frontend
cd frontend

# Instal dependensi Node.js
npm install

# Jalankan server development React
npm start

Aplikasi React akan terbuka secara otomatis di browser pada http://localhost:3000.