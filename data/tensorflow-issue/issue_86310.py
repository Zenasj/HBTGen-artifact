import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import numpy as np
import tensorflow as tf

# Set a fixed batch size
batch_size = 32

# Create some random training data
# We'll have sequences of length 5, with 1 feature per time step
sequence_length = 5
num_features = 1
num_samples = 100  # Total number of samples (must be divisible by batch_size)

# Ensure num_samples is a multiple of batch_size
num_samples = (num_samples // batch_size) * batch_size

X_train = np.random.rand(num_samples, sequence_length, num_features)
y_train = np.random.rand(num_samples, 1)  # Example target values

# Reshape y_train to match expected output shape if needed
y_train = y_train.reshape(-1,1)

# Create the stateful LSTM model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(units=64,  # Number of LSTM units
                               batch_input_shape=(batch_size, sequence_length, num_features),
                               stateful=True,
                               return_sequences=False)) #often false for a final prediction

model.add(tf.keras.layers.Dense(units=1)) # Output layer with 1 unit

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
epochs = 10

for epoch in range(epochs):
    # Shuffle data indices for each epoch (important for stateful LSTMs)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    X_train = X_train[indices]
    y_train = y_train[indices]

    model.fit(X_train, y_train, batch_size=batch_size, epochs=1, shuffle=False) # Shuffle must be false

    # Reset states after each epoch (essential for stateful LSTMs)
    model.reset_states()

