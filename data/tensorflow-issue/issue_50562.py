import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import numpy as np

data = np.random.random((12800, 120, 120, 3,))
label = np.array([1 for _ in range(12800)])
print(data)

model = tf.keras.Sequential(
    [tf.keras.layers.Conv2D(256, kernel_size=5, strides=2, input_shape=(120, 120, 3,), activation="relu"),
     tf.keras.layers.Conv2D(256, kernel_size=5, strides=2, input_shape=(120, 120, 3,), activation="relu"),
     tf.keras.layers.GlobalAvgPool2D(),
     tf.keras.layers.Dense(1, activation="sigmoid")]
)
model.compile(loss=tf.keras.losses.BinaryCrossentropy())
model.fit(x=data, y=label, batch_size=32, epochs=20)

import tensorflow as tf

@tf.function(jit_compile=True)
def recompiled_on_launch(a, b):
  return a + b

recompiled_on_launch(tf.ones([1, 10]), tf.ones([1, 10]))
recompiled_on_launch(tf.ones([1, 100]), tf.ones([1, 100]))