# tf.sparse.SparseTensor with shape=(batch_size, None) and dtype=tf.int64, batch_size=10, variable-length sequences per batch

import tensorflow as tf
import numpy as np

class SparseSlice(tf.keras.layers.Layer):
    def __init__(self, feature_column):
        super(SparseSlice, self).__init__()
        self.fc = feature_column

    def build(self, input_shape):
        # Kernel shape is number of buckets to embed from the categorical column
        self.kernel = self.add_weight(
            '{}_kernel'.format(self.fc.name),
            shape=(self.fc.num_buckets,),
            dtype=tf.float32,
            trainable=True,
            initializer='random_uniform'
        )

    def call(self, input_sparse_tensor: tf.SparseTensor):
        # FeatureColumn method to transform input tensor: outputs SparseTensor with int64 values
        ids = self.fc._transform_input_tensor(input_sparse_tensor)
        # Gather kernel weights according to SparseTensor values (indices not required here)
        gathered = tf.gather(self.kernel, ids.values)
        # Expand dims to match Dense layer expected input shape: (batch_size * variable_length, 1)
        return tf.expand_dims(gathered, axis=1)


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Use a categorical_column_with_hash_bucket with 10,000 buckets and int64 dtype consistent with input
        self.feature_name = 'smth'
        self.batch_size = 10
        self.col = tf.feature_column.categorical_column_with_hash_bucket(
            key=self.feature_name,
            hash_bucket_size=10000,
            dtype=tf.int64
        )
        # SparseSlice layer using the feature column
        self.sparse_slice = SparseSlice(self.col)
        # A final Dense layer projecting from 1 feature dim to 1 output dim with sigmoid activation
        self.dense = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        # inputs is expected to be a SparseTensor with dtype=tf.int64 and dense_shape=[batch_size, None]
        # Extract representation via SparseSlice
        x = self.sparse_slice(inputs)  # shape: (sum of all values in batch sequences, 1)
        # Pass through dense layer
        output = self.dense(x)  # shape: (sum_seq_elements, 1)
        # Note: Depending on the use-case, the output shape shape might be ragged due to sparse inputs,
        # but we keep it as is because the original snippet did so.
        return output


def my_model_function():
    # Return an instance of MyModel with weights initialized (random uniform)
    return MyModel()


def GetInput():
    # Produce a SparseTensor matching the expected input of MyModel:
    # batch_size=10, each batch element with sequence length=1 (to match example in issue),
    # dtype=tf.int64 to match feature column dtype
    batch_size = 10
    # Create indices for a sparse tensor of shape (batch_size, 1)
    indices = [[i, 0] for i in range(batch_size)]
    # Random int64 values hashed into [0, 10000)
    values = np.random.randint(0, 10000, size=(batch_size,), dtype=np.int64)
    dense_shape = [batch_size, 1]

    # Construct the sparse tensor input
    sparse_input = tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=dense_shape)
    return sparse_input

