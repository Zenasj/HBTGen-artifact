import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import numpy as np
rand = np.random.default_rng()

x = np.random.rand(1000, 200, 256, 3)
y = np.array([rand.integers(20) for i in range(1000)])

model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (200, 256, 3)),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(32, (3,3), activation = 'relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation = tf.nn.relu),
            tf.keras.layers.Dense(20, activation = tf.nn.softmax)
        ])
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x, y, batch_size=2, epochs=4)