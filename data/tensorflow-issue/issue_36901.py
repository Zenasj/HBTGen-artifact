import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import numpy as np
print(tf.__version__)

x_input = tf.keras.Input((8,8,8,1),batch_size=1)
x = tf.keras.layers.ConvLSTM2D(4,(2,2),return_sequences=True)(x_input)
x = tf.keras.layers.MaxPooling3D((1, 2, 2))(x)
x = tf.keras.layers.Flatten()(x)
x_output = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inputs=x_input,outputs=x_output)

model.compile(loss='mse',optimizer='adam')

data_in = np.random.rand(1,8,8,8,1).astype(np.float32)
data_out = np.random.rand(1,1).astype(np.float32)

r = model.fit(data_in,data_out)
print(r)