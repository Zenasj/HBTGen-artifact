from tensorflow.keras import layers

import tensorflow as tf
tf.autograph.set_verbosity(10, alsologtostdout = True)

from tensorflow.keras.layers import Layer, Input

class SlashPhobic(Layer):

    def call(self, inputs):
        s = "foo" \
            "bar"
        print(s)
        return inputs

x = Input(shape = (1,))
y = SlashPhobic()(x)