import random
from tensorflow.keras import layers
from tensorflow.keras import models

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Dropout,InputLayer


# Check TensorFlow version
print("TensorFlow version:", tf.__version__)

# Build a simple neural network model using Input layer
model = Sequential([
    InputLayer(input_shape=(20,)),  # Use Input layer to specify input shape
    Dense(64, activation='relu'),
    Dense(10)
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Create dummy data
import numpy as np
x_train = np.random.random((1000, 20))
y_train = np.random.randint(10, size=(1000,))

# Try using model.fit
try:
    model.fit(x_train, y_train, epochs=10)
    print("Model training successful, no error occurred.")
except AttributeError as e:
    print("An AttributeError occurred:", e)