import random
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import tensorflow as tf

x = np.random.random((200, 20, 20, 3)) * 10
y = x.dot(np.random.random((3, 3)))
x = x.astype(np.uint8)

model = tf.keras.Sequential()
model.add(
    tf.keras.layers.Conv2D(
        filters=3,
        kernel_size=3,
        activation="relu",
        input_shape=(20, 20, 3),
        padding="same",
    )
)
model.summary()
model.compile(optimizer="adam", loss="mse")

# Works
gen = tf.data.Dataset.from_tensor_slices((x, y)).batch(16).repeat()
model.fit(gen, epochs=2, steps_per_epoch=10)

# Works
gen3 = tf.data.Dataset.from_tensor_slices((x, y, np.ones((200,)))).batch(20).repeat()
model.fit(gen3, epochs=2, steps_per_epoch=10)

# Doesnt work
gen2 = tf.data.Dataset.from_tensor_slices((x, y, np.ones((200,)))).batch(16).repeat()
model.fit(gen2, epochs=2, steps_per_epoch=10)