from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

# %%

import tensorflow as tf
import numpy as np

data = list(map(lambda n: n / 10000, range(0, 10000+1)))

# %%

sequences = list(zip(data, data[1:], data[2:], data[3:], data[4:], data[5:]))
train_x = np.array(list(map(
    lambda seq: seq[:-1], sequences))).reshape(len(sequences), len(sequences[0])-1, 1)
train_y = np.array(
    list(map(lambda seq: seq[-1], sequences)))

model = tf.keras.Sequential([
    tf.keras.layers.CuDNNLSTM(
        3,
        return_sequences=False),
    tf.keras.layers.Dense(1)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.1, decay=0.003), 
    loss=tf.keras.losses.mse)

# %%

model.fit(train_x, train_y, epochs=100, validation_split=0, shuffle=False)