from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import numpy as np

print('Using Tensorflow version {} (git version {})'.format(tf.version.VERSION, tf.version.GIT_VERSION))

batch_size = 3
ts = 9
input_dim = 2
nump = np.arange(examples*batch_size*ts*input_dim, dtype=np.float32).reshape(batch_size, ts, input_dim)
dataset = tf.data.Dataset.from_tensor_slices(nump).batch(batch_size)
for x in dataset:
    print(x.shape)
return_state = True

model_seq = tf.keras.Sequential([tf.keras.layers.LSTM(3, return_state=return_state)])
for x in dataset:
    print(model_seq(x))

def lstm_model(return_state, ts, input_dim):
    inp = tf.keras.Input(shape=(ts, input_dim))
    out = tf.keras.layers.LSTM(3, return_state=return_state)(inp)
    return tf.keras.Model(inputs=inp, outputs=out)
    
model_func = lstm_model(return_state, ts, input_dim)

for x in dataset:
    print(model_func(x))