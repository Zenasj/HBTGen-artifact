# tf.sparse.SparseTensor with shape [None, None] and tf.Tensor with shape [None, 1]
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # A simple dense layer model mapping input of shape (None, 1) to output (None, 1)
        self.dense = tf.keras.layers.Dense(1)

    @tf.function
    def call(self, inputs):
        # Just call the dense layer on dense inputs
        # inputs are expected to be dense tensor matching shape [None, 1]
        return self.dense(inputs)

    @tf.function
    def serving_fn(self, x: tf.SparseTensor, y: tf.Tensor):
        # This function replicates the serving signature logic from the issue:
        # Convert SparseTensor input x to a dense indicator tensor with depth 5,
        # cast to int64, and return dictionary with processed 'x' and original 'y'.

        # tf.sparse.to_indicator converts sparse indices into dense indicator tensor
        x_out = tf.cast(tf.sparse.to_indicator(x, 5), tf.int64)  # Shape [batch, 5]
        return {"x": x_out, "y": y}

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Create an example SparseTensor and a dense tensor matching expected inputs for serving_fn
    # Assumptions based on the issue:
    # - 'x' is a SparseTensor of dtype int64 and shape [batch_size, variable length]
    # - 'y' is a dense int64 tensor of shape [batch_size, 1]

    # Example batch of 3 with ragged input converted to sparse:
    ragged = tf.ragged.constant([[1,3], [2,3,1], [2]], dtype=tf.int64)
    x_sparse = ragged.to_sparse()  # SparseTensor shape [3, max_length=3]

    y = tf.expand_dims(tf.constant([1, 2, 1], dtype=tf.int64), axis=1)  # shape [3,1]

    return (x_sparse, y)

