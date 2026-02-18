from fastapi import FastAPI, Request
import numpy as np
import tensorflow as tf
import joblib
from collections import deque
import logging

# Configure logging for Render dashboard visibility
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. Create FastAPI app instance
app = FastAPI()

# 2. Load the trained model, scaler, and threshold
#    Ensure these files exist in the same directory on Render
try:
    model = tf.keras.models.load_model('lstm_autoencoder.keras')
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise e

try:
    scaler = joblib.load('scaler.save')
    logger.info("Scaler loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load scaler: {e}")
    raise e

try:
    # threshold.npy contains a single float value
    THRESHOLD = np.load('threshold.npy').item()
    logger.info(f"Threshold loaded: {THRESHOLD}")
except Exception as e:
    logger.error(f"Failed to load threshold, using default 0.08: {e}")
    THRESHOLD = 0.08   # fallback only if file missing

# 3. Sliding window buffer (holds last 20 readings)
#    Each reading is a list: [current, temperature, vibration]
data_buffer = deque(maxlen=20)

# 4. Define the prediction endpoint
@app.post("/")
async def predict(request: Request):
    """
    Accepts a JSON with current, temperature, vibration.
    Returns anomaly flag, MSE, and status.
    """
    try:
        # Parse incoming JSON
        content = await request.json()
        curr = float(content['current'])
        temp = float(content['temperature'])
        vibe = float(content['vibration'])

        # Add to buffer
        data_buffer.append([curr, temp, vibe])

        # Not enough data yet
        if len(data_buffer) < 20:
            return {
                "is_anomaly": False,
                "status": "collecting",
                "message": f"Need {20 - len(data_buffer)} more readings"
            }

        # Prepare sequence: (20, 3)
        sequence = np.array(data_buffer)           # shape (20, 3)
        # Scale using the fitted scaler
        scaled_seq = scaler.transform(sequence)    # shape (20, 3)
        # Reshape for LSTM: (1, 20, 3)
        input_tensor = scaled_seq.reshape(1, 20, 3).astype('float32')

        # Run inference (reconstruction)
        reconstructed = model.predict(input_tensor, verbose=0)   # shape (1, 20, 3)

        # Compute MSE (mean over all time steps and features)
        mse = np.mean(np.square(input_tensor - reconstructed))

        # Compare with threshold
        is_anomaly = bool(mse > THRESHOLD)

        return {
            "is_anomaly": is_anomaly,
            "mse": round(float(mse), 6),
            "threshold": float(THRESHOLD),
            "status": "ready"
        }

    except KeyError as e:
        # Missing field in JSON
        logger.warning(f"Missing key in request: {e}")
        return {"error": f"Missing field: {e}"}
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"error": str(e)}

# Optional health check endpoint (useful for Render monitoring)
@app.get("/health")
async def health():
    return {"status": "healthy"}