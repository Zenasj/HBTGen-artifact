from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import tensorflow_hub as hub

class MyModel(tf.keras.Model):    
    def __init__(self):
        super().__init__()
        self.embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        self.dense = tf.keras.layers.Dense(1)
        
    def call(self, inputs, training=False):
        x = self.embed(inputs)
        x = self.dense(x)
        return x
    
model = MyModel()
x = ['a sentence', 'b sentence']
y = [0, 1]

model(x, y)  # works fine

model.compile(loss='mse', optimizer='sgd')
model.fit(x, y)  # errors

import tensorflow as tf
import tensorflow_hub as hub

# adapted from https://github.com/tensorflow/hub/issues/648
hub_layer = hub.KerasLayer('https://tfhub.dev/google/universal-sentence-encoder/4', output_shape=(None, 512), input_shape=(None,), trainable=True, dtype=tf.string)
inputs = tf.keras.layers.Input(shape=(None,), dtype='string')

hub_layer(tf.squeeze(inputs))  # works
hub_layer(inputs)  # errors