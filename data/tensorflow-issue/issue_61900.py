from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import tensorflow as tf

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.activations import linear
from tensorflow.python.keras.optimizers import adam_v2
from tensorflow.python.keras.losses import MeanSquaredError

x = [ [23], [45], [78], [12]]
y = [ [12], [22], [36], [6]]

model = Sequential([
    tf.keras.layers.Dense(units=25, activation='relu', name='layer1'),
    tf.keras.layers.Dense(input_shape=(25,), units=15, activation='relu', name='layer2'),
    tf.keras.layers.Dense(input_shape=(15,), units=1, activation='linear', name='layer3')
] ,name="Model1"
)

loss=MeanSquaredError()
opt=adam_v2.Adam(learning_rate=0.001)

model.compile(
    loss=loss,
    optimizer=opt
    )
model.fit(x, y, epochs=100)