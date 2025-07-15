import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # Import train_test_split
from sklearn.preprocessing import StandardScaler
import kagglehub
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import shap
import joblib

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
model.save('model.h5')
print("Model saved to model.h5")

# --- 🔹 Plot Loss ---
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid(True)
plt.show()