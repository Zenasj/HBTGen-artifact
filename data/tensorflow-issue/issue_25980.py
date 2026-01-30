from tensorflow.keras import layers

import tensorflow as tf
from tensorflow import keras
print(tf.__version__)
header_input = keras.Input(shape=(10,), name='header', sparse=True, batch_size=10)
header_features = keras.layers.Reshape((1, 10))(header_input)

tf.keras.layers.Lambda(tf.sparse.to_dense)(x)

def sparse_to_dense(value: Any):
    if isinstance(value, tf.sparse.SparseTensor):
        return tf.sparse.to_dense(value)
    return value

tf.keras.layers.Lambda(sparse_to_dense)(x)

class SparseToDense(tf.keras.layers.Layer):
    def __init__(self):
        super(SparseToDense, self).__init__()

    def build(self, input_shape):
        pass

    def call(self, input: Any) -> object:
        if isinstance(input, tf.sparse.SparseTensor):
            return tf.sparse.to_dense(input)
        return input