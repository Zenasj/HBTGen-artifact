from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import numpy as np

seq_len = 2
batch_size = 1
feature_dim = 1

input = tf.keras.Input(shape=(seq_len, feature_dim))
# Transpose input to be time major
input_transposed = tf.transpose(input, perm=[1,0,2])
output = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(1, return_sequences=True, time_major=True), name='bi')(input_transposed)
model = tf.keras.Model(inputs=input, outputs=output)

# Set all the weights to be one for simplicity
rnn_layer = model.get_layer('bi')
weights = rnn_layer.get_weights()
new_w = [np.ones(x, dtype=np.float32) for x in [(feature_dim, 4), (1, 4), (4)] * 2]
rnn_layer.set_weights(new_w)

model.save("test.h5")
x = np.ones((batch_size, seq_len, feature_dim), dtype=np.float32)
expected = model.predict(x)
print(expected)