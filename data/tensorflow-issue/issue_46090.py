import random
from tensorflow.keras import layers

import tensorflow as tf
import numpy as np
class CustomModel(keras.Model):
    def __init__(self, hidden_units):
        super(CustomModel, self).__init__()
        self.dense_layers = [keras.layers.Dense(u) for u in hidden_units]

    def call(self, x, y):
        x = tf.concat((x, y), axis=1)
        for layer in self.dense_layers:
            x = layer(x)
        return {'x': x}


model = CustomModel([16, 16, 10])
# Build the model by calling it
input_arr = tf.random.uniform((1, 5))
input_arr_2 = tf.random.uniform((1, 10))
outputs = model(input_arr, input_arr_2)
print(outputs)
model.save("my_model")