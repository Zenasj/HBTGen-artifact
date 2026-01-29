# tf.random.uniform((B, 100), dtype=tf.float32) ← assumed input shape and dtype for demonstration

import tensorflow as tf

class SparseToDense(tf.keras.layers.Layer):
    """
    A custom Keras layer to convert sparse tensor inputs to dense tensors at runtime.
    
    This layer addresses the issue that Keras Dropout (and other layers) do not support
    sparse tensors as input directly, resulting in type conversion errors.
    
    Using this layer allows feeding sparse inputs to the model while converting them
    to dense just before layers like Dropout.
    
    Note: Converting from sparse to dense at runtime may increase memory usage and runtime,
    especially for very large sparse inputs. Use cautiously.
    """
    def call(self, inputs):
        if tf.sparse.is_sparse(inputs):
            return tf.sparse.to_dense(inputs)
        return inputs


class MyModel(tf.keras.Model):
    """
    A model demonstrating handling of sparse inputs with Dropout layer,
    by converting sparse inputs to dense inside the model.

    Input: a sparse tensor of shape (batch_size, 100)
    Processing:
        - Converts sparse input to dense
        - Applies Dropout(0.2)
        - Dense layer with 64 units and ReLU activation
        - Output Dense layer with 10 units (for example classification)
    """
    def __init__(self):
        super().__init__()
        # Layer to convert sparse input to dense
        self.sparse_to_dense = SparseToDense()

        # Dropout layer
        self.dropout = tf.keras.layers.Dropout(0.2)

        # Dense layers after dropout
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.out = tf.keras.layers.Dense(10)  # e.g., 10-class output

    @tf.function(jit_compile=True)  # Enable XLA compilation
    def call(self, inputs, training=False):
        # inputs is expected to be sparse tensor batch of shape (batch_size, 100)
        x = self.sparse_to_dense(inputs)
        x = self.dropout(x, training=training)
        x = self.dense1(x)
        return self.out(x)


def my_model_function():
    """
    Returns an instance of MyModel.
    """
    return MyModel()

def GetInput():
    """
    Returns a sparse tensor input matching the expected input of MyModel:
    a batch of sparse vectors with shape (batch_size, 100), dtype float32.

    Here, we simulate a sparse batch with random indices and values.
    """

    batch_size = 8
    feature_dim = 100
    # Suppose each sample has ~10 nonzero entries on average
    num_nonzero_per_sample = 10
    total_nonzero = batch_size * num_nonzero_per_sample

    # Generate random batch indices for sparse values
    batch_indices = tf.repeat(tf.range(batch_size), num_nonzero_per_sample)  # shape=(total_nonzero,)
    # Generate random feature indices per nonzero element
    feature_indices = tf.random.uniform(shape=(total_nonzero,), maxval=feature_dim, dtype=tf.int64)

    # Combine to create indices of shape (total_nonzero, 2)
    indices = tf.stack([batch_indices, feature_indices], axis=1)

    # Random float32 values for sparse elements
    values = tf.random.uniform(shape=(total_nonzero,), dtype=tf.float32)

    # Dense shape of the sparse tensor
    dense_shape = [batch_size, feature_dim]

    sparse_input = tf.sparse.SparseTensor(indices=indices,
                                         values=values,
                                         dense_shape=dense_shape)
    # Optionally reorder and prune duplicates if any (recommended)
    sparse_input = tf.sparse.reorder(sparse_input)

    return sparse_input

