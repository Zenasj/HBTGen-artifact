import random
from tensorflow import keras

import tensorflow as tf
from keras import layers
import numpy as np

input_layer = layers.Input(shape=(500, 5))
input_lstm = layers.LSTM(30, return_sequences=True)(input_layer)
output1 = layers.Dense(1)(input_lstm)
output2 = layers.Dense(1)(input_lstm)

model = tf.keras.Model(inputs=input_layer, outputs=[output1, output2])

model.compile(optimizer="adam", run_eagerly=False, sample_weight_mode='temporal', loss="mse", metrics=[["mae"], ["mae"]])


#dataset
x = np.random.random((2000, 500, 5))

sample_weights = np.ones(x.shape[:-1])
amnt_zeros = np.random.choice(500, 2000)
for idx, zeros in enumerate(amnt_zeros):
    sample_weights[idx, -zeros:] = 0.0

x = x*sample_weights[...,None]
y1 = ((np.sum(x, axis=-1) + 20) * sample_weights)[..., None]
y2 = ((np.sum(x, axis=-1) + 10) * sample_weights)[...,None]


#masked y3 data is increased drasically to show the wrong calculation of the metrics
y2_testsample_weights = np.full_like(y2, -50000) * ((sample_weights - 1)[...,None])
y2 = y2 + y2_testsample_weights

history = model.fit(x=x, y=[y1, y2], sample_weight=sample_weights, epochs=500)