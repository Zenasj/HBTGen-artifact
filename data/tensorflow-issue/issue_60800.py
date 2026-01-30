import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
import numpy as np

# Generate random training data
np.random.seed(0)
x_train = np.random.rand(100, 1)
y_train = 3 * x_train + 2 + np.random.randn(100, 1) * 0.1

# Define the neural network architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,))
])

# Compile the model
model.compile(optimizer='sgd', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, epochs=1000, batch_size=10)

### Relevant log output