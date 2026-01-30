from tensorflow.keras import layers

import tensorflow as tf
from tensorflow import keras

# Define the shape of the input tensor
input_shape = (128, 128, 1)

# Create an input tensor with the specified shape
input_tensor = keras.layers.Input(shape=input_shape)

# Create a model that uses the input tensor
x = keras.layers.Conv2D(32, 3, activation='relu', padding='same')(input_tensor)
x = keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
output_tensor = keras.layers.Conv2D(1, 3, activation='sigmoid', padding='same')(x)

print(type(output_tensor)) # <class 'keras.engine.keras_tensor.KerasTensor'>

# Convert the output tensor to a numpy array
print(type(output_tensor.numpy())) # AttributeError: 'KerasTensor' object has no attribute 'numpy'
print(type(tf.keras.backend.eval(output_tensor))) #AttributeError: 'KerasTensor' object has no attribute 'numpy'`