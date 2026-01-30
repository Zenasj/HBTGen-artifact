import random
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import tensorflow as tf

FEATURES_DIM = 5
_randoms = np.random.random((10, 32, FEATURES_DIM))

# # This is my wish: ragged input to sequence feature column
# features = tf.RaggedTensor.from_tensor(_randoms, ragged_rank=1)
# labels = tf.reduce_sum(tf.expand_dims(features, axis=-2), axis=-1)

# This is what should work right now
_indexes = tf.where(tf.not_equal(_randoms, 0.0))
_values = tf.gather_nd(_randoms, _indexes)
features = tf.SparseTensor(_indexes, _values, _randoms.shape)
labels = tf.sparse.reduce_sum(features, axis=-1, keepdims=True)    

dataset = tf.data.Dataset \
    .from_tensor_slices(({'features': features}, labels)) \
    .batch(32)


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.features = tf.keras.experimental.SequenceFeatures([
            tf.feature_column.sequence_numeric_column('features', shape=(FEATURES_DIM,))
        ])
        self.dense_1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(32, activation='relu'))
        self.dense_2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))

    def call(self, inputs):
        outputs = self.features(inputs)
        outputs = self.dense_1(outputs)
        outputs = self.dense_2(outputs)
        
        return outputs

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = 1
        
        return tf.TensorShape(shape)

    
model = MyModel()
model.compile(optimizer='Adam', loss='mse')
model.fit_generator(dataset, epochs=5)

class M(tf.keras.Model):
  def __init__(self):
    super(M, self).__init__()
    ...
    self._features = tf.keras.layers.DenseFeatures([...])

  def call(self, inputs):
    # inputs is a dict of sparse tensors.
    layer = self._features(inputs)
    ...
    return prediction

import numpy as np
import tensorflow as tf

FEATURES_DIM = 5
_randoms = np.random.random((10, 32, FEATURES_DIM))

# # This is my wish: ragged input to sequence feature column
# features = tf.RaggedTensor.from_tensor(_randoms, ragged_rank=1)
# labels = tf.reduce_sum(tf.expand_dims(features, axis=-2), axis=-1)

# This is what should work right now
_indexes = tf.where(tf.not_equal(_randoms, 0.0))
_values = tf.gather_nd(_randoms, _indexes)
features = tf.SparseTensor(_indexes, _values, _randoms.shape)
labels = tf.sparse.reduce_sum(features, axis=-1, keepdims=True)    

dataset = tf.data.Dataset \
    .from_tensor_slices(({'features': features}, labels)) \
    .batch(32)


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.features = tf.keras.experimental.SequenceFeatures([
            tf.feature_column.sequence_numeric_column('features', shape=(FEATURES_DIM,))
        ])
        self.dense_1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(32, activation='relu'))
        self.dense_2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))

    def call(self, inputs):
        outputs = self.features(inputs)
        outputs = self.dense_1(outputs[0])
        outputs = self.dense_2(outputs)
        
        return outputs

    
model = MyModel()
model.compile(optimizer='Adam', loss='mse')
model.fit_generator(dataset, epochs=5)