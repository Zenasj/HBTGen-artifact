# tf.random.uniform((16, 2048, 1028), dtype=tf.float32) ← inferred input shape from JAX ShapeDtypeStruct x

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # The original code defines a flax.linen Dense layer with 1024 output units and a custom dot_general operator.
        # We'll implement a Keras Dense equivalent without the custom dot_general since it is JAX-specific.
        # Output dimension 1024 inferred from nn.Dense(1024,...)
        self.dense = tf.keras.layers.Dense(
            1024,
            activation=None,
            use_bias=True,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros'
        )
    
    def call(self, inputs, training=False):
        # inputs shape: (16, 2048, 1028) - as per ShapeDtypeStruct in the JAX example
        # We apply the dense layer to the last dimension.
        # This replicates a batched Dense application.
        # In TensorFlow: dense applies to last axis by default.
        x = self.dense(inputs)
        return x

def my_model_function():
    # Instantiate and return the model
    return MyModel()

def GetInput():
    # Return a random tensor input matching the expected input:
    # Shape: (16, 2048, 1028), dtype: float32, matching the JAX example shape
    return tf.random.uniform((16, 2048, 1028), dtype=tf.float32)

