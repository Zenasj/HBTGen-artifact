import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# --- Configuration ---
INPUT_DIM = 10
OUTPUT_DIM = 1

# --- Model Definition ---
input_layer = layers.Input(shape=(INPUT_DIM,), name='input_features', dtype=tf.float32)
hidden = layers.Dense(8, activation='relu', name='hidden_layer')(input_layer)
output_layer = layers.Dense(OUTPUT_DIM, activation='linear', name='output_prediction')(hidden)
model = keras.Model(inputs=input_layer, outputs=output_layer, name="simple_sequential_model")
model.compile(optimizer='adam', loss='mse')

# Deterministic input data
input_data = np.arange(INPUT_DIM, dtype=np.float32).reshape(1, INPUT_DIM)
input_data = (input_data / INPUT_DIM) - 0.5

# GPU Inference
gpu_pred = model.predict(input_data)
print(f"GPU prediction: {gpu_pred}")

# CPU Inference
with tf.device('/CPU:0'):
    cpu_pred = model.predict(input_data)
print(f"CPU prediction: {cpu_pred}")

