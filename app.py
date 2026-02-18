import numpy as np
from collections import deque

# 1. Setup (Run this once)
data_buffer = deque(maxlen=20)
THRESHOLD = 0.08 

# 2. Simulation Function
def test_single_input(curr, temp, vibe):
    # Add new data to the window
    data_buffer.append([curr, temp, vibe])
    
    if len(data_buffer) < 20:
        print(f"Collecting... ({len(data_buffer)}/20)")
        return
    
    # Process exactly like the Render API will
    np_buffer = np.array(data_buffer)
    scaled_buffer = scaler.transform(np_buffer) # Uses your trained scaler
    input_ready = scaled_buffer.reshape(1, 20, 3).astype('float32')
    
    reconstructed = model.predict(input_ready, verbose=0)
    mse = np.mean(np.power(input_ready - reconstructed, 2))
    
    print(f"MSE: {mse:.6f} | Anomaly: {mse > THRESHOLD}")

# 3. Try it out (Run this multiple times to fill the buffer)
test_single_input(2.22, 65.6, 2.99)