# tf.sparse.SparseTensor with shape (batch, heads, query_length, key_length) and
# dense tensor with shape (batch, heads, key_length, value_dim)

import tensorflow as tf

def Dense2Sparse():
    # This function is implemented as a Keras Model that converts a dense tensor and mask to a SparseTensor
    dense = tf.keras.Input((None, None, None))  # shape: (batch, heads, query_length, key_length)
    mask = tf.keras.Input((1, None, None))  # shape: (batch, 1, query_length or 1, key_length)

    # Conditionally tile mask to match dense's shape in query_length dimension
    reshaped_mask = tf.keras.layers.Lambda(
        lambda x: tf.cond(
            tf.math.not_equal(tf.shape(x[0])[2], tf.shape(x[1])[2]),
            lambda: tf.tile(x[0], [1, 1, tf.shape(x[1])[2], 1]),
            lambda: x[0],
        )
    )([mask, dense])  # shape: (batch, 1, query_length, key_length)

    # Tile mask to match number of heads
    reshaped_mask = tf.keras.layers.Lambda(
        lambda x: tf.tile(x[0], [1, tf.shape(x[1])[1], 1, 1])
    )([reshaped_mask, dense])  # shape: (batch, heads, query_length, key_length)

    # Get indices where mask is True / non-zero
    indices = tf.keras.layers.Lambda(lambda x: tf.where(tf.cast(x, dtype=tf.int32)))(reshaped_mask)
    # Gather values from dense at these indices
    values = tf.keras.layers.Lambda(lambda x: tf.gather_nd(x[0], x[1]))([dense, indices])
    # Create SparseTensor from indices and values, with dense shape as dense tensor's shape cast to int64
    sparse = tf.keras.layers.Lambda(
        lambda x: tf.sparse.SparseTensor(x[0], values=x[1], dense_shape=tf.cast(tf.shape(x[2]), dtype=tf.int64))
    )([indices, values, dense])

    return tf.keras.Model(inputs=(dense, mask), outputs=sparse)


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # No trainable weights, only operations
        
    def call(self, inputs):
        # inputs is a tuple (a, b)
        # a is a SparseTensor with shape (batch, heads, query_length, key_length)
        # b is a dense tensor with shape (batch, heads, key_length, value_dim)

        a, b = inputs

        # Reshape sparse tensor to shape (batch * heads, query_length, key_length)
        reshaped_a = tf.sparse.reshape(a, (-1, tf.shape(a)[-2], tf.shape(a)[-1]))

        # Reshape dense tensor to shape (batch * heads, key_length, value_dim)
        reshaped_b = tf.reshape(b, (-1, tf.shape(b)[-2], tf.shape(b)[-1]))

        # Define a function that performs sparse_dense_matmul for each element pair
        def dot(x):
            sparse_mat, dense_mat = x[0], x[1]  # sparse_mat: SparseTensor (query_length, key_length), dense_mat: Tensor (key_length, value_dim)
            c = tf.sparse.sparse_dense_matmul(sparse_mat, dense_mat)  # shape: (query_length, value_dim)
            return c

        # Note: The main issue here was the fn_output_signature containing tf.shape(...)
        # which returned a tf.Tensor, not an integer dimension.
        # So we use the static known shapes from the SparseTensor and dense tensor.

        # Retrieve static shapes for signature if available
        q_len = reshaped_a.shape[-2]
        v_dim = reshaped_b.shape[-1]

        # If static shape is None, fall back to dynamic - this may cause issues,
        # but it's the common compromise.
        # For compatibility, we try static shape first:
        if q_len is None or v_dim is None:
            q_len = tf.shape(reshaped_a)[-2]
            v_dim = tf.shape(reshaped_b)[-1]

        # Use tf.map_fn over zipped sparse and dense tensors 
        # The output shape of each call is (query_length, value_dim)
        results = tf.map_fn(
            dot,
            elems=(reshaped_a, reshaped_b),
            fn_output_signature=tf.TensorSpec((q_len, v_dim), dtype=tf.float32),
        )

        # The results shape will be (batch * heads, query_length, value_dim)
        # Reshape back to (batch, heads, query_length, value_dim)
        batch = tf.shape(a)[0]
        heads = tf.shape(a)[1]
        results = tf.reshape(results, (batch, heads, tf.shape(results)[-2], tf.shape(results)[-1]))

        return results


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Generate input tensors compatible with MyModel
    batch = 4
    heads = 3
    query_length = 10
    key_length = 40
    value_dim = 20

    # Create dense input tensor (batch, heads, query_length, key_length)
    dense = tf.random.uniform((batch, heads, query_length, key_length), dtype=tf.float32)

    # Create mask tensor (batch, 1, query_length, key_length) with 0 or 1 values
    mask = tf.cast(tf.random.uniform((batch, 1, query_length, key_length), maxval=2, dtype=tf.int32), tf.bool)

    # Convert dense + mask to sparse tensor using the predefined Dense2Sparse model
    dense2sparse_model = Dense2Sparse()
    sparse_a = dense2sparse_model((dense, mask))

    # Dense tensor b (batch, heads, key_length, value_dim)
    dense_b = tf.random.uniform((batch, heads, key_length, value_dim), dtype=tf.float32)

    # Return tuple (sparse_a, dense_b) as input to MyModel
    return sparse_a, dense_b

