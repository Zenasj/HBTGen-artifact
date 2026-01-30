import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import numpy as np

class SparseSlice(tf.keras.layers.Layer):
    def __init__(self, feature_column):
        super(SparseSlice, self).__init__()
        self.fc = feature_column

    def build(self, input_shape):

        self.kernel = self.add_weight('{}_kernel'.format(self.fc.name), shape=(self.fc.num_buckets, ), dtype=tf.float32)

    def call(self, input):
        ids = self.fc._transform_input_tensor(input)
        return tf.expand_dims(tf.gather(self.kernel, ids.values), axis=1)


strategy = tf.distribute.MirroredStrategy()
with strategy.scope():

    batch_size = 10
    sparse_col = tf.feature_column.categorical_column_with_hash_bucket('sparse_col', 10000, dtype=tf.int64)
    dense_col = tf.feature_column.numeric_column('dense_col', dtype=tf.float32)
    example_spec = tf.feature_column.make_parse_example_spec([sparse_col, dense_col])

    sparse_inputs = tf.keras.layers.Input(name=sparse_col.name, shape=(None, ), batch_size=batch_size, sparse=True, dtype=tf.int64)
    dense_inputs = {dense_col.name: tf.keras.layers.Input(name=dense_col.name, shape=(1, ), dtype=tf.float32)}

    sparse_out = SparseSlice(sparse_col)(sparse_inputs)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(sparse_out)
    num = tf.keras.layers.DenseFeatures(dense_col)(dense_inputs)

    concats = tf.keras.layers.Concatenate()([output, num])
    output = tf.keras.layers.Dense(1, activation='sigmoid')(concats)

    model = tf.keras.Model([dense_inputs, {'sparse_output': sparse_inputs}], output)

    model.compile(optimizer='adam',
                  loss='mse')

    np.random.random(())

    features = {dense_col.name: tf.constant(np.random.random((batch_size, )))}
    features.update({sparse_col.name: tf.sparse.SparseTensor(indices=[[i, 0] for i in range(batch_size)], values=np.random.randint(0, 1000, (batch_size, )), dense_shape=(batch_size, 1))})
    ys = tf.constant(np.random.rand(batch_size), dtype=tf.float32)

    dataset = tf.data.Dataset.from_tensor_slices((features, ys)).batch(batch_size)

    model.fit(x=dataset,
              epochs=1
              )

strategy = tf.distribute.MirroredStrategy()
strategy.extended.experimental_enable_get_next_as_optional = False