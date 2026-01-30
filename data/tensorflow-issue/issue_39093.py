from tensorflow.keras import layers

import tensorflow as tf

import numpy as np
from tensorflow.python.ops import math_ops
from tensorflow.python import keras

class MyLayer(keras.layers.Layer):

    def call(self, inputs, training=None):
        # Expecting training to be set
        if training is not None:
            self.add_loss(math_ops.reduce_sum(inputs))

        return inputs


inputs = keras.Input((3,))
layer = MyLayer()
outputs = layer(inputs)
model = keras.Model(inputs, outputs)
model.compile('sgd', 'mse', run_eagerly=False)
loss = model.fit(np.ones((2, 3)), np.ones((2, 3)))

print(loss.history)

import tensorflow as tf
tf.enable_eager_execution()

import numpy as np
from tensorflow.python.ops import math_ops
from tensorflow.python import keras

class MyLayer(keras.layers.Layer):

    def call(self, inputs, training=None):
        # Expecting training to be set
        if training is not None:
            self.add_loss(math_ops.reduce_sum(inputs))

        return inputs


inputs = keras.Input((3,))
layer = MyLayer()
outputs = layer(inputs)
model = keras.Model(inputs, outputs)
model.compile('sgd', 'mse', run_eagerly=True)
loss = model.fit(np.ones((2, 3)), np.ones((2, 3)))

print(loss.history)

# Print out is "6" as training branch works.