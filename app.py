import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

# Load model dan pipeline
model = tf.keras.models.load_model('model.h5')
scaler = joblib.load('scaler.pkl')
le_dict = joblib.load('label_encoders.pkl')

# UI
st.title("Prediksi Harga Mobil Ford Bekas")

# Input dari user
year = st.number_input("Tahun Mobil", min_value=1990, max_value=2025, value=2015)
mileage = st.number_input("Mileage (dalam mil)", min_value=0, value=50000)
tax = st.number_input("Tax ($)", min_value=0, value=150)
mpg = st.number_input("MPG", min_value=0.0, value=40.0)
engineSize = st.number_input("Engine Size (L)", min_value=0.0, value=1.6)

model_input = st.selectbox("Model Mobil", le_dict['model'].classes_)
transmission_input = st.selectbox("Transmisi", le_dict['transmission'].classes_)
fuel_input = st.selectbox("Jenis Bahan Bakar", le_dict['fuelType'].classes_)

if st.button("Prediksi Harga"):
    # Encode input
    model_encoded = le_dict['model'].transform([model_input])[0]
    transmission_encoded = le_dict['transmission'].transform([transmission_input])[0]
    fuel_encoded = le_dict['fuelType'].transform([fuel_input])[0]

    # Gabungkan semua input
    X_input = np.array([[year, mileage, tax, mpg, engineSize,
                         model_encoded, transmission_encoded, fuel_encoded]])

    # Normalisasi
    X_scaled = scaler.transform(X_input)

    # Prediksi
    prediction = model.predict(X_scaled)[0][0]

    st.success(f"Prediksi Harga Mobil: ${prediction:,.2f}")
