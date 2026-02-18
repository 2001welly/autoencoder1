from fastapi import FastAPI, Request
import numpy as np
import tensorflow as tf
import joblib
from collections import deque

# 1. This MUST be named 'app' so Uvicorn can find it
app = FastAPI()

# 2. Load Model and Scaler
# Make sure these files are in the same folder as app.py on GitHub
model = tf.keras.models.load_model('autoencoder_model.h5')
scaler = joblib.load('scaler.pkl')

# 3. Sliding window buffer
data_buffer = deque(maxlen=20)
THRESHOLD = 0.08 

@app.post("/")
async def predict(request: Request):
    try:
        content = await request.json()
        curr = float(content['current'])
        temp = float(content['temperature'])
        vibe = float(content['vibration'])

        data_buffer.append([curr, temp, vibe])

        # If we don't have 20 readings yet, we can't run the model
        if len(data_buffer) < 20:
            return {"is_anomaly": False, "status": "collecting"}

        # Prepare data for LSTM Autoencoder (Batch, TimeSteps, Features)
        np_buffer = np.array(data_buffer)
        scaled_buffer = scaler.transform(np_buffer)
        input_ready = scaled_buffer.reshape(1, 20, 3).astype('float32')
        
        # Run inference
        reconstructed = model.predict(input_ready, verbose=0)
        mse = np.mean(np.power(input_ready - reconstructed, 2))
        
        return {
            "is_anomaly": bool(mse > THRESHOLD),
            "mse": round(float(mse), 6),
            "status": "ready"
        }
    except Exception as e:
        return {"error": str(e)}