import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf
import numpy as np

batch_size = 33

input_tensor = tf.keras.Input(shape=[5], batch_size=batch_size)
lstm_input = tf.keras.layers.Dense(16, activation=tf.keras.activations.elu)(input_tensor)
lstm_input = tf.expand_dims(lstm_input, axis=1)
lstm_layer = tf.keras.layers.LSTM(16, stateful=True,return_state=True)
lstm_out, h, c = lstm_layer(lstm_input)
out = tf.keras.layers.Dense(1, activation=tf.keras.activations.linear)(lstm_out)

model = tf.keras.Model(
    inputs=[input_tensor],
    outputs=[out, h, c])
model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError())
model.summary()

input_arr = np.random.random(size=(batch_size, 5))
model.predict(input_arr,batch_size=batch_size)