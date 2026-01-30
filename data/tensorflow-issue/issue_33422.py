import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.backend import to_dense

test_input = Input((10, 5), sparse=True)
dense_net = Lambda(to_dense, output_shape=(10, 5))(test_input)
test_net = Dense(50)(dense_net)

from tensorflow.keras.layers import Input, Dense, Lambda, Layer, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.backend import to_dense

test_input = Input((10, 5), sparse=True)
dense_net = Lambda(to_dense)(test_input)
reshape_net = Reshape((10, 5))(dense_net)
test_net = Dense(50)(reshape_net)

class ToDenseLayer(Layer):
    def __init__(self, out_shape):
        super(ToDenseLayer, self).__init__()
        self.out_shape = out_shape

    def call(self, inputs, **kwargs):
        return tf.sparse.to_dense(inputs)

    def compute_output_shape(self, input_shape):
        return None, None, self.out_shape

class ToDenseLayer(Layer):
    def __init__(self, out_shape):
        super(ToDenseLayer, self).__init__()
        self.out_shape = out_shape

    def call(self, inputs, **kwargs):
        dense_tensor = tf.sparse.to_dense(inputs)
        return tf.ensure_shape(dense_tensor, [None, None, self.out_shape])