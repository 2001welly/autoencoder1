from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from collections import deque
import os

app = FastAPI()

# Load artifacts at startup
model = load_model("lstm_autoencoder.keras")
scaler = joblib.load("scaler.save")
threshold = np.load("threshold.npy").item()   # if saved as .npy
# If you saved as .txt, use: threshold = float(open("threshold.txt").read())

window_size = 20

# Buffer to hold recent readings (in‑memory; for production, consider Redis)
buffer = deque(maxlen=window_size)

class SensorReading(BaseModel):
    current: float
    temperature: float
    vibration: float

@app.get("/")
def root():
    return {"message": "Motor Anomaly Detection API is running"}

@app.post("/predict")
def predict(reading: SensorReading):
    # Convert reading to array
    point = np.array([[reading.current, reading.temperature, reading.vibration]])
    
    # Add to buffer
    buffer.append(point.flatten())
    
    # If buffer not yet full, inform client
    if len(buffer) < window_size:
        return {
            "status": "buffering",
            "message": f"Collecting data. Need {window_size - len(buffer)} more readings."
        }
    
    # Create sequence from buffer (last window_size points)
    sequence = np.array(buffer)  # shape (20, 3)
    
    # Scale using the fitted scaler
    seq_scaled = scaler.transform(sequence)
    
    # Reshape for model: (1, 20, 3)
    input_seq = np.expand_dims(seq_scaled, axis=0)
    
    # Reconstruct
    reconstructed = model.predict(input_seq, verbose=0)
    
    # Compute MSE
    mse = np.mean(np.square(input_seq - reconstructed))
    
    # Compare with threshold
    anomaly = bool(mse > threshold)
    
    return {
        "anomaly": anomaly,
        "mse": float(mse),
        "threshold": float(threshold)
    }