import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

# Load scaler dan label encoder
scaler = joblib.load("scaler.pkl")
le_dict = joblib.load("label_encoders.pkl")

# Load model tflite
interpreter = tf.lite.Interpreter(model_path="modelcar_price_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Judul Aplikasi
st.title("🚘 Prediksi Harga Mobil Ford")
st.write("Masukkan spesifikasi mobil:")

# Ambil opsi dari encoder
model_options = list(le_dict["model"].classes_)
transmission_options = list(le_dict["transmission"].classes_)
fuel_options = list(le_dict["fuelType"].classes_)

year = st.number_input("Tahun", min_value=1990, max_value=2025, value=2017)
mileage = st.number_input("Jarak tempuh (dalam mil)", min_value=0, value=40000)
tax = st.number_input("Pajak ($)", min_value=0, value=150)
mpg = st.number_input("MPG", min_value=0.0, value=55.4)
engineSize = st.number_input("Ukuran Mesin", min_value=0.0, value=1.6)
model_input = st.selectbox("Model", model_options)
transmission_input = st.selectbox("Transmisi", transmission_options)
fuel_input = st.selectbox("Tipe Bahan Bakar", fuel_options)

if st.button("Prediksi Harga"):
    # Encode
    model_encoded = le_dict['model'].transform([model_input])[0]
    transmission_encoded = le_dict['transmission'].transform([transmission_input])[0]
    fuel_encoded = le_dict['fuelType'].transform([fuel_input])[0]

    data = np.array([[year, mileage, tax, mpg, engineSize,
                      model_encoded, transmission_encoded, fuel_encoded]])

    # Scale
    data_scaled = scaler.transform(data)

    # Predict with TFLite
    interpreter.set_tensor(input_details[0]['index'], data_scaled.astype(np.float32))
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    st.success(f"Prediksi harga mobil: ${prediction[0][0]:,.2f}")
