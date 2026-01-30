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


batch_size = 10
c = 'smth'
col = tf.feature_column.categorical_column_with_hash_bucket(c, 10000, dtype=tf.int64)
example_spec = tf.feature_column.make_parse_example_spec([col])

inputs = tf.keras.layers.Input(name=c, shape=(None, ), batch_size=batch_size, sparse=True, dtype=tf.int64)
sparse_out = SparseSlice(col)(inputs)
output = tf.keras.layers.Dense(1, activation='sigmoid')(sparse_out)

model = tf.keras.Model(inputs, output)

model.compile(optimizer='adam',
              loss='mse')


features = {c: tf.sparse.SparseTensor(indices=[[i, 0] for i in range(batch_size)], values=np.random.randint(0, 1000, (batch_size, )).tolist(), dense_shape=(batch_size, 1))}
ys = tf.constant(np.random.rand(batch_size).tolist(), dtype=tf.float32)

dataset = tf.data.Dataset.from_tensor_slices((features, ys)).batch(batch_size)

model.fit(x=dataset,
          epochs=1
          )

model.predict({c: np.random.randint(0, 1000, (batch_size, 1))})