from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from collections import deque
import joblib  # To load your scaler

app = Flask(__name__)

# 1. Load your trained model
model = tf.keras.models.load_model('autoencoder_model.h5')

# 2. Load the scaler you used during training
# IMPORTANT: You must upload your scaler.pkl to GitHub/Render 
# so the AI uses the same math as the training phase.
try:
    scaler = joblib.load('scaler.pkl')
except:
    # Fallback if scaler isn't found (highly recommended to use your actual scaler)
    scaler = None 

# 3. Create a sliding window buffer for 20 readings
# Each reading has 3 features: [current, temperature, vibration]
data_buffer = deque(maxlen=20)

# 4. Define your Anomaly Threshold
# Adjust this based on your Colab testing (e.g., 0.05 or 0.1)
THRESHOLD = 0.08 

@app.route('/', methods=['POST'])
def predict():
    try:
        # Get data from ESP32
        content = request.json
        curr = float(content['current'])
        temp = float(content['temperature'])
        vibe = float(content['vibration'])

        # Add to sliding window
        new_reading = [curr, temp, vibe]
        data_buffer.append(new_reading)

        # Check if we have enough readings to satisfy the model (20)
        if len(data_buffer) < 20:
            return jsonify({
                "is_anomaly": False, 
                "status": f"Collecting data: {len(data_buffer)}/20"
            }), 200

        # --- PREDICTION LOGIC ---
        
        # Convert buffer to numpy array
        np_buffer = np.array(data_buffer) # Shape: (20, 3)

        # Scale the data (Crucial!)
        if scaler:
            scaled_buffer = scaler.transform(np_buffer)
        else:
            scaled_buffer = np_buffer

        # Reshape for model: (Batch, TimeSteps, Features) -> (1, 20, 3)
        input_ready = scaled_buffer.reshape(1, 20, 3).astype('float32')

        # Predict/Reconstruct
        reconstructed = model.predict(input_ready, verbose=0)

        # Calculate Mean Squared Error
        mse = np.mean(np.power(input_ready - reconstructed, 2))

        # Determine if it's an anomaly
        is_anomaly = bool(mse > THRESHOLD)

        return jsonify({
            "is_anomaly": is_anomaly,
            "mse": round(float(mse), 6),
            "status": "OK"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)