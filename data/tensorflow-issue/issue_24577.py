import random
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import tensorflow as tf
from tensorflow import keras
import numpy as np

X_train = np.random.rand(1000, 2).astype(np.float32)
y_train = np.random.rand(1000).astype(np.float32)
X_valid = np.random.rand(200, 2).astype(np.float32)
y_valid = np.random.rand(200).astype(np.float32)
batch_size = 32
learning_rate = 0.01

train_set = tf.data.Dataset.from_tensor_slices((X_train, y_train)).repeat().batch(batch_size)
valid_set = tf.data.Dataset.from_tensor_slices((X_valid, y_valid)).repeat().batch(batch_size)

model = keras.models.Sequential([keras.layers.Dense(1)])
model.compile(loss="mse", optimizer=keras.optimizers.SGD(learning_rate))

model.fit(train_set, epochs=5,
          steps_per_epoch=len(X_train) // batch_size,
          validation_data=valid_set,
          validation_steps=len(X_valid) // batch_size)