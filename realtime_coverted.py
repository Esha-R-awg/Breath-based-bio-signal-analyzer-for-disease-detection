import serial
import numpy as np
import joblib
import tensorflow as tf
import time
from tensorflow.lite.python.interpreter import Interpreter

# === Load TFLite Model ===
interpreter = Interpreter(model_path="C:/Users/dhars/OneDrive/Breath_based_project/model_converted.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# === Load Scaler and Label Encoder ===
scaler = joblib.load("C:/Users/dhars/OneDrive/Breath_based_project/scaler_final_1.pkl")
label_encoder = joblib.load("C:/Users/dhars/OneDrive/Breath_based_project/label_encoder_final_1.pkl")

# === Setup Serial ===
ser = serial.Serial('COM6', 9600)  # Check your COM port
time.sleep(2)  # Wait for ESP32 to initialize

print("Calibrating sensors...")

# === Calibration Phase ===
calibration_samples = []
num_calibration = 10

while len(calibration_samples) < num_calibration:
    try:
        line = ser.readline().decode().strip()
        print(f"Calib Reading: {line}")
        values = line.split(",")
        if len(values) == 4:
            MQ3, MQ135, Temp, Hum = map(float, values)
            calibration_samples.append([MQ3, MQ135, Temp, Hum])
    except:
        continue

# Compute calibration baseline
calibration_array = np.array(calibration_samples)
baseline = np.mean(calibration_array, axis=0)
print(f"Calibration complete! Baseline: {baseline}")

print("\n--- Live prediction started ---\n")

# === Live Prediction Loop ===
while True:
    try:
        line = ser.readline().decode().strip()
        print("========================================")
        print(f"Raw Data: {line}")

        if line:
            values = line.split(",")
            if len(values) == 4:
                MQ3, MQ135, Temp, Hum = map(float, values)

                # Subtract baseline if necessary (optional)
                adjusted = np.array([[MQ3, MQ135, Temp, Hum]])  # or: - baseline

                # Scale input
                input_scaled = scaler.transform(adjusted).astype(np.float32)

                # Inference
                interpreter.set_tensor(input_details[0]['index'], input_scaled)
                interpreter.invoke()
                output = interpreter.get_tensor(output_details[0]['index'])
                predicted_index = np.argmax(output)
                predicted_label = label_encoder.inverse_transform([predicted_index])[0]

                print(f" Predicted Disease: {predicted_label}")

                # === Send prediction back to ESP32 for OLED ===
                ser.write((predicted_label + "\n").encode('utf-8'))

    except Exception as e:
        print(f"Error: {e}")
        continue
