import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import time

model = tf.keras.Sequential()
layers = 15
for _ in range(layers):
    model.add(tf.keras.layers.Conv2D(3, 3, data_format="channels_first"))
    
model_input = tf.random.uniform(shape=(12, 3, 256, 256))
xla_fn = tf.function(model, jit_compile=True)

iterations = 100

for i in range(iterations):
    start_time = time.perf_counter()  # time.time() is not suitable for benchmarking
    model_out = xla_fn(model_input)
    tf.test.experimental.sync_devices()  # Force GPU resync point - Sync about it as "waiting for the pipeline of async ops to clear out"
    end_time = time.perf_counter()
    print(f"Iteration {i} time: {1000 * (end_time - start_time)}")