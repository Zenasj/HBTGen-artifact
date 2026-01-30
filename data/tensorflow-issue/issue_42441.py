from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import numpy as np
import tensorflow as tf

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

for i in range(50):
    tf.print(f'Model {i}...')

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=2))
    model.add(tf.keras.layers.Dense(units=1))

    prediction = model.predict(x)