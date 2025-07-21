import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # Import train_test_split
from sklearn.preprocessing import StandardScaler
import kagglehub
from tensorflow.keras.models import Sequential, load_model # type: ignore
from tensorflow.keras.layers import Dense, InputLayer, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
import shap
import joblib
import json

# --- 🔹 Utility Functions for Model Loading ---
def load_trained_model(model_path='crop_yield_model.keras', scaler_path='x_scaler.pkl'):
    """
    Load trained model and scaler for easy prediction
    """
    try:
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        print(f"✅ Model loaded from {model_path}")
        print(f"✅ Scaler loaded from {scaler_path}")
        return model, scaler
    except Exception as e:
        print(f"❌ Error loading model/scaler: {e}")
        return None, None

def predict_yield(model, scaler, features):
    """
    Make prediction with loaded model
    Args:
        model: loaded keras model
        scaler: loaded StandardScaler
        features: list of features [Temperature_C, Rainfall_mm, Humidity_percent, 
                                   Fertilizer_kg_per_hectare, Pesticide_kg_per_hectare, 
                                   Soil_quality_index]
    """
    try:
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0][0]
        return float(prediction)
    except Exception as e:
        print(f"❌ Error making prediction: {e}")
        return None

# --- 🔹 Data Loading ---
# No need to download every time if it's cached
try:
    yield_dataset = pd.read_csv('/root/.cache/kagglehub/datasets/blueloki/synthetic-agricultural-yield-prediction-dataset/versions/1/agricultural_yield_train.csv')
    yield_test = pd.read_csv('/root/.cache/kagglehub/datasets/blueloki/synthetic-agricultural-yield-prediction-dataset/versions/1/agricultural_yield_test.csv')
except FileNotFoundError:
    print("Downloading dataset...")
    path = kagglehub.dataset_download("blueloki/synthetic-agricultural-yield-prediction-dataset")
    yield_dataset = pd.read_csv(f"{path}/agricultural_yield_train.csv")
    yield_test = pd.read_csv(f"{path}/agricultural_yield_test.csv")

print(yield_dataset.head())
print(yield_test.head())
yield_dataset = pd.read_csv(path + '/agricultural_yield_train.csv')
print(yield_dataset.head())
# yield_dataset.info()       # Tipe data, jumlah data, missing
# yield_dataset.shape        # Jumlah baris dan kolom
# yield_dataset.isnull().sum()



yield_test = pd.read_csv(path + '/agricultural_yield_test.csv')
print(yield_test.head())
# yield_test.info()       # Tipe data, jumlah data, missing
# yield_test.shape        # Jumlah baris dan kolom
# yield_test.isnull().sum()
# --- 🔹 IDA: Initial Data Analysis ---
print("\n📊 DATA INFO:")
print(yield_dataset.info())

print("\n🔍 Missing Values:")
print(yield_dataset.isnull().sum())

print("\n📈 Descriptive Stats:")
print(yield_dataset.describe())

# --- 🔹 Histogram Distribusi Setiap Fitur ---
import seaborn as sns
num_cols = yield_dataset.select_dtypes(include=[np.number]).columns

yield_dataset[num_cols].hist(figsize=(15, 10), bins=30)
plt.suptitle("Distribusi Fitur", fontsize=16)
plt.tight_layout()
plt.show()

# --- 🔹 Heatmap Korelasi ---
plt.figure(figsize=(10, 8))
sns.heatmap(yield_dataset[num_cols].corr(), annot=True, cmap="coolwarm")
plt.title("Korelasi Antar Fitur")
plt.show()

# --- 🔹 Skewness Check ---
print("\n📐 Skewness Sebelum Transformasi:")
print(yield_dataset[num_cols].skew())

# Log Transform fitur yang terlalu skewed
# from scipy.stats import skew

# skewed_feats = yield_dataset[num_cols].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
# high_skew = skewed_feats[skewed_feats > 1]
# print("\n🔍 Fitur Skewed (>1):")
# print(high_skew)

# for col in high_skew.index:
#     yield_dataset[col] = np.log1p(yield_dataset[col])
#     print(f"✅ Log transform applied on: {col}")

# # Visualisasi ulang setelah log transform
# yield_dataset[high_skew.index].hist(figsize=(12, 8), bins=30)
# plt.suptitle("Distribusi Setelah Log Transform", fontsize=16)
# plt.tight_layout()
# plt.show()






# Feature Engineering
yield_dataset['log_Fertilizer_Amount_kg_per_hectare'] = np.log1p(yield_dataset['Fertilizer_Amount_kg_per_hectare'])
yield_dataset['log_Rainfall_mm'] = np.log1p(yield_dataset['Rainfall_mm'])
yield_dataset['log_Irrigation_Schedule'] = np.log1p(yield_dataset['Irrigation_Schedule'])
yield_test['log_Fertilizer_Amount_kg_per_hectare'] = np.log1p(yield_test['Fertilizer_Amount_kg_per_hectare'])
yield_test['log_Rainfall_mm'] = np.log1p(yield_test['Rainfall_mm'])
yield_test['log_Irrigation_Schedule'] = np.log1p(yield_test['Irrigation_Schedule'])
# 🔹 Fitur 1: Efektivitas Pemupukan
# yield_dataset['Fertilizer_per_Soil'] = yield_dataset['log_Fertilizer_Amount_kg_per_hectare'] / yield_dataset['Soil_Quality']
# yield_test['Fertilizer_per_Soil'] = yield_test['log_Fertilizer_Amount_kg_per_hectare'] / yield_test['Soil_Quality']

# 🔹 Fitur 2: Total Air (Curah Hujan + Irigasi)
yield_dataset['Total_Water'] = yield_dataset['log_Rainfall_mm'] + yield_dataset['log_Irrigation_Schedule'] * 50
yield_test['Total_Water'] = yield_test['log_Rainfall_mm'] + yield_test['log_Irrigation_Schedule'] * 50

# 🔹 Fitur 3: Efisiensi Cahaya terhadap Hasil Panen
yield_dataset['Light_per_Yield'] = yield_dataset['Sunny_Days'] / (yield_dataset['Yield_kg_per_hectare'] + 1)
yield_test['Light_per_Yield'] = yield_test['Sunny_Days'] / (yield_test['Yield_kg_per_hectare'] + 1)

# 🔹 Fitur 4: Interaksi Varietas Benih dan Irigasi
yield_dataset['Seed_Irrigation_Interaction'] = yield_dataset['Seed_Variety'] * yield_dataset['log_Irrigation_Schedule']
yield_test['Seed_Irrigation_Interaction'] = yield_test['Seed_Variety'] * yield_test['log_Irrigation_Schedule']

# # 🔹 Fitur 5: Interaksi Kualitas Tanah dan Sinar Matahari
# yield_dataset['Soil_Sun_Interaction'] = yield_dataset['Soil_Quality'] * yield_dataset['Sunny_Days']
# yield_test['Soil_Sun_Interaction'] = yield_test['Soil_Quality'] * yield_test['Sunny_Days']


# Buat subset kolom fitur baru + target
cols = ['Yield_kg_per_hectare',  'Total_Water', 'Light_per_Yield',
        'Seed_Irrigation_Interaction', ]

# Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(yield_dataset[cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Korelasi Fitur Baru terhadap Target")
plt.show()

# --- 🔹 Creating X and y variables ---
# Define features (X) and target (y)
# We assume 'Yield' is the target. Drop it from the features.
X_train = yield_dataset.drop('Yield_kg_per_hectare', axis=1)

y_train = yield_dataset['Yield_kg_per_hectare']


X_test = yield_test.drop('Yield_kg_per_hectare', axis=1)
y_test = yield_test['Yield_kg_per_hectare']
# Create training and testing sets from the loaded data
# The test CSV provided doesn't have a 'Yield' column, so we'll split the training data
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

print(f"\nCreated training set with shape: {X_train.shape}")
print(f"Created testing set with shape: {X_test.shape}")


# --- 🔹 Preprocessing ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Simpan scaler yang sudah di-fit agar bisa digunakan oleh API
joblib.dump(scaler, 'x_scaler.pkl')
print("Scaler saved to x_scaler.pkl")
# -------------------------------------------

# --- 🔹 Model Building ---
model = Sequential([
    InputLayer(input_shape=(X_train_scaled.shape[1],)),
    Dense(128, activation='relu'),
    Dropout(0.1),
    Dense(64, activation='relu'),
    Dropout(0.1),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse'
)

# --- 🔹 Callbacks ---
early_stop = EarlyStopping(
    patience=20,
    restore_best_weights=True,
    monitor='val_loss'
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=1e-6,
    verbose=1
)

# --- 🔹 Training ---
history = model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr],
    verbose=1,
    shuffle=True
)

# --- 🔹 Evaluation ---
loss = model.evaluate(X_test_scaled, y_test)
print(f"\nTest Loss (MSE): {loss:.5f}")

# --- 🔹 Save Model dengan Multiple Format ---
# 1. Save dalam format .keras (recommended untuk TensorFlow 2.x)
model.save('crop_yield_model.keras')
print("Model saved to crop_yield_model.keras")

# 2. Save dalam format .h5 (untuk kompatibilitas)
model.save('model.h5')
print("Model saved to model.h5")

# 3. Save model summary ke text file
with open('model_summary.txt', 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))
print("Model summary saved to model_summary.txt")

# 4. Save model architecture ke JSON
model_json = model.to_json()
with open('model_architecture.json', 'w') as f:
    f.write(model_json)
print("Model architecture saved to model_architecture.json")

# 5. Save hanya weights
model.save_weights('model_weights.h5')
print("Model weights saved to model_weights.h5")

# --- 🔹 Plot Loss ---
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

# --- 🔹 Demo: Test Model Loading ---
print("\n" + "="*50)
print("🧪 TESTING MODEL LOADING")
print("="*50)

# Test loading the saved model
loaded_model, loaded_scaler = load_trained_model()

if loaded_model and loaded_scaler:
    # Test prediction dengan sample data
    sample_features = [25.0, 150.0, 70.0, 50.0, 5.0, 80.0]  # Temperature, Rainfall, Humidity, Fertilizer, Pesticide, Soil_quality
    
    print(f"\n📊 Sample Input Features:")
    feature_names = ['Temperature_C', 'Rainfall_mm', 'Humidity_percent', 
                    'Fertilizer_kg_per_hectare', 'Pesticide_kg_per_hectare', 'Soil_quality_index']
    for name, value in zip(feature_names, sample_features):
        print(f"  {name}: {value}")
    
    predicted_yield = predict_yield(loaded_model, loaded_scaler, sample_features)
    
    if predicted_yield:
        print(f"\n🌾 Predicted Yield: {predicted_yield:.2f} kg/hectare")
        print("✅ Model loading and prediction successful!")
    else:
        print("❌ Prediction failed!")
else:
    print("❌ Model loading failed!")

print("\n" + "="*50)
print("📁 SAVED FILES:")
print("="*50)
print("1. crop_yield_model.keras - Full model (recommended)")
print("2. model.h5 - Legacy format")
print("3. x_scaler.pkl - Feature scaler")
print("4. model_summary.txt - Model architecture summary")
print("5. model_architecture.json - Model structure")
print("6. model_weights.h5 - Model weights only")
print("\n🚀 Ready for deployment!")
