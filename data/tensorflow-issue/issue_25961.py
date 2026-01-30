import tensorflow as tf

import collections

from tensorflow.python.ops import math_ops
from tensorflow.python.feature_column.feature_column import _CategoricalColumn


class _DebuggingCategoricalColumn(
    _CategoricalColumn,
    collections.namedtuple('_DebuggingCategoricalColumn', (
        'dense_col',
    ))):

    @property
    def name(self):
        return 'debugging_weighted'

    @property
    def _parse_example_spec(self):
        config = self.dense_col._parse_example_spec  # pylint: disable=protected-access
        return config

    @property
    def _num_buckets(self):
        return 4

    def _transform_feature(self, inputs):
        v = inputs.get(self.dense_col)
        batch_size = tf.to_int64(tf.shape(v)[0])
        buckets = tf.to_int64(tf.squeeze(math_ops.bucketize(input=v, boundaries=(0, 2, 4))))
        indices = tf.stack([
            tf.range(batch_size, dtype=tf.int64),
            tf.zeros(batch_size, dtype=tf.int64)], axis=1)
        id_tensor = tf.SparseTensor(
            indices=indices,
            values=tf.to_int64(tf.squeeze(buckets)),
            dense_shape=[batch_size, 1]
        )
        id_weights = tf.SparseTensor(
            indices=indices,
            values=tf.ones(batch_size, dtype=tf.float32),
            dense_shape=[batch_size, 1]
        )
        return id_tensor, id_weights

    def _get_sparse_tensors(self, inputs, weight_collections=None, trainable=None):
        tensors = inputs.get(self)
        return _CategoricalColumn.IdWeightPair(tensors[0], tensors[1])


def debugging_column(input_column):
    return _DebuggingCategoricalColumn(dense_col=input_column)


df = pd.DataFrame({'x': [-1., 1, 3, 5], 'y': [1., 1, 2, 3]})
feat = tf.feature_column.numeric_column('x')
feature_columns = [debugging_column(feat)]
input_fn = tf.estimator.inputs.pandas_input_fn(df, df['y'], shuffle=True, num_epochs=None)
estimator = tf.estimator.LinearRegressor(feature_columns=feature_columns)
estimator.train(input_fn=input_fn, steps=100)