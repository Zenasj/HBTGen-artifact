# tf.random.uniform((23, 29), dtype=tf.float32) ← inferred input shape and type from example

import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    A reimplementation of the LayerNormalization custom layer shown in the issue,
    adapted for TensorFlow 2.x style (tf.keras.Model) for compatibility with TF 2.20.
    
    Note:
    - The original code uses tf.layers.Layer and tf.get_variable with variable_scope,
      which is TF 1.x style.
    - This implementation uses tf.keras.layers.Layer with self.add_weight for variables.
    - The input shape is arbitrary, but for this minimal example, (batch, features) = (23, 29).
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # Initialize scale and bias as trainable variables
        self.scale = self.add_weight(
            name="layer_norm_scale",
            shape=(self.hidden_size,),
            initializer=tf.ones_initializer(),
            trainable=True
        )
        self.bias = self.add_weight(
            name="layer_norm_bias",
            shape=(self.hidden_size,),
            initializer=tf.zeros_initializer(),
            trainable=True
        )

    def call(self, x, epsilon=1e-6):
        # Compute mean and variance along last dimension
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=-1, keepdims=True)
        norm_x = (x - mean) * tf.math.rsqrt(variance + epsilon)
        # Apply scale and bias and return normalized output
        return norm_x * self.scale + self.bias


def my_model_function():
    # Create an instance of MyModel with hidden_size = 29
    # (matches the last dimension of input tensor in GetInput)
    return MyModel(hidden_size=29)

def GetInput():
    # Return a random tensor input matching expected shape and dtype
    # Using the shape from the original example: (23, 29)
    # dtype float32 as per tf.random_uniform default in TF 1.x (now tf.random.uniform)
    return tf.random.uniform((23, 29), dtype=tf.float32)

