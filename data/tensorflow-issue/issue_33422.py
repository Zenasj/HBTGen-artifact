# tf.random.uniform((B, 10, 5), dtype=tf.float32) ← input shape inferred from Input((10, 5), sparse=True)

import tensorflow as tf

class ToDenseLayer(tf.keras.layers.Layer):
    def __init__(self, out_dim):
        super(ToDenseLayer, self).__init__()
        self.out_dim = out_dim  # expected last dimension size, e.g., 5

    def call(self, inputs, **kwargs):
        # Convert sparse input to dense tensor
        dense_tensor = tf.sparse.to_dense(inputs)
        # Ensure tensor shape is properly set (None=batch, None=sequence length, out_dim=feature dim)
        return tf.ensure_shape(dense_tensor, [None, None, self.out_dim])

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Custom to_dense conversion layer to handle sparse input and provide proper shape info
        self.to_dense_layer = ToDenseLayer(out_dim=5)
        # Following Dense layer that requires known last dimension in input shape
        self.dense = tf.keras.layers.Dense(50)

    def call(self, inputs):
        x = self.to_dense_layer(inputs)
        x = self.dense(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a sparse tensor input that matches the model's expected input shape: (batch, 10, 5)
    # Since tf.sparse.to_dense expects a SparseTensor input, simulate a sparse tensor.
    # For demonstration, we'll create dense tensor and convert to sparse
    batch_size = 4  # arbitrary batch size
    dense_input = tf.random.uniform((batch_size, 10, 5), dtype=tf.float32)
    # Convert dense tensor to sparse tensor
    sparse_input = tf.sparse.from_dense(dense_input)
    return sparse_input

