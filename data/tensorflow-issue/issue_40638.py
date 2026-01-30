import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.keras.utils.plot_model(
    model, to_file='model.png', show_shapes=False, show_layer_names=True,
    rankdir='TB', expand_nested=False, dpi=96
)

class transformer_IO(tf.keras.layers.Layer):
    def call(self, input):
        return (input, None, None, None)

class transformer_IO_model(tf.keras.Model):
    def call(self, input):
        return (input, None, None, None)

from tensorflow.python.keras.engine import keras_tensor
keras_tensor.enable_keras_tensors()