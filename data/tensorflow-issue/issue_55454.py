# tf.random.uniform((B, 1), dtype=tf.float64) ← Inferred input shape based on keras.Input(shape=(1), dtype='float64')

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define the layers with dtype float64 as per the reported issue's example
        self.dense1 = tf.keras.layers.Dense(4, activation="relu", dtype=tf.float64)
        self.dense2 = tf.keras.layers.Dense(2, dtype=tf.float64)

    def call(self, inputs):
        # Forward pass as per original code:
        # some_input -> Dense(4, relu) -> Dense(2) -> then replaced using tf.where with a constant tensor
        x = self.dense1(inputs)
        x = self.dense2(x)

        # Mask is tf.equal([0,1], 1) from the example, which is [False, True]
        # The constants used in tf.where are float64 tensors.
        mask = tf.equal([0, 1], 1)  # shape (2,), bool tensor: [False, True]

        # Constant tensor with the same dtype float64 and shape compatible with x
        other_value = tf.constant([0.1], dtype=tf.float64)

        # tf.where(condition, x, y)
        # The issue described occurs when x and y have different dtypes or incompatible shapes.
        # According to the workaround, keeping dtypes consistent avoids the problem.
        # Here, 'x' argument is other_value (shape (1,)), 'y' is x (shape (None,2)) from dense output.

        # To ensure compatible shapes, broadcast other_value to shape of x,
        # here batching dim is unknown (None), so explicitly broadcast
        # other_value has shape (1,), x has shape (batch_size, 2)
        # broadcast other_value to (batch_size, 2) with tf.broadcast_to
        # Since mask shape is (2,), and tf.where matches mask shape to last dims,
        # mask broadcasts across batch.

        # Broadcast the constant to batch size unknown dynamically
        batch_size = tf.shape(x)[0]
        other_value_broadcasted = tf.broadcast_to(other_value, [batch_size, 2])

        # tf.where will choose elements according to mask per element dimension:
        replaced_value = tf.where(mask, other_value_broadcasted, x)
        return replaced_value


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Return a random input tensor matching keras.Input(shape=(1), dtype='float64')
    # Batch size is arbitrary, choose 2 for demonstration
    batch_size = 2
    return tf.random.uniform((batch_size,1), dtype=tf.float64)

