from tensorflow import keras
from tensorflow.keras import layers

from osgeo import gdal

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Create a simple model
model = Sequential([
    Dense(64, activation='relu', input_shape=(32,)),
    Dense(10)
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Print the model summary
model.summary()

# Save the initial weights to a file
model.save_weights('my_model_weights.h5')

# Change the weights of the model
for layer in model.layers:
    weights = layer.get_weights()  # list of numpy arrays
    weights = [weight * 0 for weight in weights]
    layer.set_weights(weights)

# Check that weights have been changed
print('Weights after resetting:')
print(model.layers[0].get_weights())

# Load the initial weights from the file
model.load_weights('my_model_weights.h5')

# Check that weights have been loaded
print('Weights after loading from file:')
print(model.layers[0].get_weights())