# tf.random.uniform((B, 2), dtype=tf.float32) ← Input is a batch of vectors with shape (batch_size, 2)

import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    This model demonstrates how to include a trainable variable separate from inputs,
    following the approach discussed in the GitHub issue. The class implements a custom
    layer that has a trainable bias independent of the inputs.

    Input shape: (batch_size, 2)
    Output shape: (batch_size, 1)
    The output ignores the direct inputs and produces a trainable bias vector per sample.
    """

    def __init__(self, output_dim=1):
        super().__init__()
        self.output_dim = output_dim
        # Initialize the bias as a trainable weight with shape (1,)
        # This simulates a global scalar bias added to every output.
        self.bias = self.add_weight(
            shape=(self.output_dim,),
            initializer="zeros",
            dtype=tf.float32,
            trainable=True,
            name="bias"
        )

    def call(self, inputs):
        """
        Forward pass:
        - inputs: Tensor with shape (batch_size, 2)
        - Compute the mean over the last axis to produce a shape (batch_size, 1)
        - Then multiply by zero to discard input effect,
          finally add the trainable bias broadcasted over batch dimension.
        This ensures output shape is (batch_size, output_dim) independent of input values.
        """
        batch_size = tf.shape(inputs)[0]
        # reduce_mean produces shape (batch_size, 1)
        mean_reduce = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        # zero out input effect
        zeros = mean_reduce * 0.0
        # add bias broadcasted to batch dimension
        output = zeros + self.bias
        # Ensure output shape is (batch_size, output_dim)
        return output

def my_model_function():
    # Return an instance of MyModel with default output dimension 1
    return MyModel()

def GetInput():
    # Return a random tensor input matching expected input: shape (batch_size, 2), float32 dtype
    # Using batch size 8 as a reasonable example
    batch_size = 8
    input_shape = (batch_size, 2)
    return tf.random.uniform(input_shape, dtype=tf.float32)

