import random

import tensorflow as tf
from tensorflow.keras import layers, models

# Define a simple model
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(100,)),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Generate some random input data
import numpy as np
x_train = np.random.random((1000, 100))
y_train = np.random.randint(10, size=(1000,))

# Train the model
model.fit(x_train, y_train, epochs=5)

import tensorflow as tf

# Set TensorFlow to use only the CPU
tf.config.set_visible_devices([], 'GPU')

from tensorflow.keras import backend as K

# Clear the Keras session to reset the model's state
K.clear_session()

import tensorflow as tf

# List available GPUs
physical_devices = tf.config.list_physical_devices('GPU')

# Set memory growth on GPU to avoid TensorFlow allocating all GPU memory upfront
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

# Check that memory growth is enabled
for device in physical_devices:
    print(f"Memory growth enabled for {device}")