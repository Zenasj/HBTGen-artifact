import math
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import tensorflow as tf
print(tf.__version__)

def create_model(use_mask: bool) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=[None], dtype=tf.int32)
    if use_mask:
        mask = inputs >= 0
    else:
        mask = None
    h = tf.keras.layers.Embedding(64, 2048)(tf.math.maximum(inputs, 0))
    h1 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(2048, return_sequences=True), merge_mode="sum")(h, mask=mask)
    h2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(2048, return_sequences=True), merge_mode="sum")(h, mask=mask)
    h = tf.keras.layers.Dense(10)(h1 + h2)
    return tf.keras.Model(inputs, h)

# Data
data = tf.data.Dataset.from_tensor_slices([[j if j <= i else -1 for j in range(64)] for i in range(64)]).batch(64)

# However, when masking is used, even prediction on GPU gives different result.
# It also sometimes crases with the error `CUDA_ERROR_ILLEGAL_ADDRESS`.
# The full error log is copied below.
# If `use_mask=False` is passed, no problem happens.

# Models
tf.keras.utils.set_random_seed(42)
model = create_model(use_mask=True)

# Run prediction
gold = None
for i in range(100):
    result = model.predict(data, verbose=0)
    if gold is None:
        gold = result
    print("Batch {}, max difference {}, mean difference {}".format(i, np.max(np.abs(gold - result)), np.mean(np.abs(gold - result))))