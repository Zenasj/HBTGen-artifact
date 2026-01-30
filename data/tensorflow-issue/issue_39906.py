import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class CustomModel(tf.keras.Model):
    def __init__(self, hidden_units):
        super(CustomModel, self).__init__()
        self.dense_layers = [
            tf.keras.layers.Dense(u) for u in hidden_units]
    def call(self, inputs):
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        return x

model = CustomModel([16, 16, 10])
# Build the model by calling it
input_arr = tf.random.uniform((1, 5))
outputs = model(input_arr)
model.save('my_custom_model')

model = CustomModel([16, 16, 10])
input_arr = tf.random.uniform((1, 5))
outputs = model(input_arr)
model._set_inputs(input_arr) # add this line
model.save('my_custom_model')

eps = tf.random.normal(shape=mean.shape)

eps = tf.random.normal(shape=tf.shape(mean))