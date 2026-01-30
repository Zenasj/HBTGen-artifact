from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
print(tf.version.GIT_VERSION, tf.version.VERSION, flush=True)
print(tf.config.list_physical_devices(), flush=True)


class CustomLayer(tf.keras.layers.Layer):
    def __init__(self, name=None, **kwargs):
        super(CustomLayer, self).__init__(name=name, **kwargs)
        self.conv_1 = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1))
        self.conv_2 = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1))

    def call(self, inputs):
        output_1 = self.conv_1(inputs)
        output_2 = self.conv_2(inputs)

        return output_1, output_2

    def compute_output_shape(self, input_shape):
        output_shape_1 = self.conv_1.compute_output_shape(input_shape)
        output_shape_2 = self.conv_2.compute_output_shape(input_shape)

        return output_shape_1, output_shape_2

try:
    inputs = tf.keras.Input(shape=(None, None, None, 1))

    custom_layer = CustomLayer()
    output_1, output_2 = tf.compat.v1.keras.layers.TimeDistributed(custom_layer)(inputs)
except Exception as e:
    print("Failed! Error:", str(e), flush=True)
else:
    print("Success!", flush=True)