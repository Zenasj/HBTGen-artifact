from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import numpy as np

seq_len = 5
vocab_size = 10
batch_size = 2
hidden_size = 20

seq_input = tf.keras.layers.Input(seq_len, dtype=tf.int32)
embs = tf.keras.layers.Embedding(vocab_size, hidden_size, mask_zero=True)
rnn = tf.keras.layers.GRU(hidden_size)
out = rnn(embs(seq_input))

keras_model = tf.keras.Model(
    inputs=seq_input,
    outputs=out)

working_input = np.array(
    [[0, 1, 2, 3, 4], # if this sequence has a pad, CuDNN is happy
     [0, 0, 0, 0, 0]])

broken_input = np.array(
    [[1, 2, 3, 4, 5], # if this sequence doesn't have a pad, CuDNN is sad
     [0, 0, 0, 0, 0]])

print('Computing the working input:')
res1 = keras_model.predict(working_input)
print(res1.shape)
print(res1[1])
print('Computing the broken input:')
# this line causes the CuDNN error if running on GPU
res2 = keras_model.predict(broken_input)
print(res2.shape)
print(res2[1])

# By default, this will pad using 0s; it is configurable via the
# "value" parameter.
# Note that you could "pre" padding (at the beginning) or
# "post" padding (at the end).
# We recommend using "post" padding when working with RNN layers
# (in order to be able to use the 
# CuDNN implementation of the layers).
padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(raw_inputs,
                                                              padding='post')