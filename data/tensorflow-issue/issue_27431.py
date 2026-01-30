from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

class Layer(tf.keras.layers.Layer):

    def __init__(self):
        super(Layer, self).__init__()
        self.layer_fn = tf.keras.layers.Dense

layer = Layer()
print(layer.variables)