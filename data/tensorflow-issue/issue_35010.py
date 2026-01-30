import random
from tensorflow import keras
from tensorflow.keras import layers

import psutil
import numpy as np
import tensorflow as tf


class TestLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        output_dim,
        **kwargs
    ):
        super(TestLayer, self).__init__(**kwargs)
        self.output_dim = output_dim

    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        batch_embedding = tf.py_function(
            self.mock_output, inp=[inputs], Tout=tf.float64,
        )
        return batch_embedding

    def mock_output(self, inputs):
        shape = inputs.shape.as_list()
        batch_size = shape[0]
        return tf.constant(np.random.random((batch_size,self.output_dim)))


test_layer = TestLayer(1000)

for i in range(1000):
    test_layer.call(tf.constant(np.random.randint(0,100,(256,10))))
    if i % 100 == 0:
        used_mem = psutil.virtual_memory().used
        print('used memory: {} Mb'.format(used_mem / 1024 / 1024))

@tf.function
def call(self, inputs):
    batch_embedding = tf.py_function(
        self.mock_output, inp=[inputs], Tout=tf.float64,
    )
    return batch_embedding

tf.reset_default_graph()
tf.keras.backend.clear_session()