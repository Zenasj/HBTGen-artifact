from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
from tensorflow import keras
import numpy as np

X = np.arange(6).astype(np.float32).reshape(-1, 1)
y = X**2
dataset = tf.data.Dataset.from_tensor_slices((X,y))
dataset = dataset.shuffle(100, reshuffle_each_iteration=True)
dataset = dataset.repeat(2)
dataset = dataset.batch(2)

@tf.function
def log_inputs(inputs):
    tf.print(inputs)
    return inputs

model = keras.models.Sequential([
    keras.layers.Lambda(log_inputs),
    keras.layers.Dense(1)
])
model.compile(loss="mse", optimizer="sgd")
model.fit(dataset, epochs=2, verbose=0)

[[5]   # first epoch, first iteration, first batch
 [2]]
[[3]      # second batch
 [1]]
[[0]      # third batch
 [4]]
[[0]   # first epoch, second iteration, first batch
 [3]]
[[1]      # second batch
 [5]]
[[4]      # third batch
 [2]]
[[5]   # second epoch, first iteration, first batch
 [2]]
[[3]
 [1]]
[[0]
 [4]]
[[0]   # second epoch, second iteration, first batch
 [3]]
[[1]
 [5]]
[[4]
 [2]]