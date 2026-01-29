# tf.random.uniform((B, 5), dtype=tf.float32) ← Input shape inferred from example: batch size B=1, feature size=5

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, seed_a, seed_b):
        super().__init__()
        self.seed_a = seed_a
        self.seed_b = seed_b

    @tf.function(jit_compile=True)
    def call(self, inputs):
        # Inputs shape: (B, 5), dtype float32 assumed from example

        # To reproduce the issue described:
        # - Two calls to tf.random.uniform with the same input shape and dtype,
        #   but possibly identical or different seeds seed_a and seed_b
        # - Multiply by an all-ones tensor to keep shape the same as input

        x = inputs / inputs  # == 1.0, shape preserved

        # Generate two random tensors using different seeds as per constructor args
        a = tf.random.uniform(tf.shape(x), dtype=x.dtype, minval=-5, maxval=5, seed=self.seed_a) * x
        b = tf.random.uniform(tf.shape(x), dtype=x.dtype, minval=-5, maxval=5, seed=self.seed_b) * x

        # Return both outputs as a list, matching the original example
        return [a, b]

def my_model_function(seed_a=123, seed_b=124):
    # Return an instance of MyModel initialized with given seeds,
    # default seeds chosen to reflect the example where seeds differ
    return MyModel(seed_a, seed_b)

def GetInput():
    # Return a random input tensor matching expected input (batch=1, features=5, float32)
    # The original example used np.ones with shape (1,5), dtype float32
    # Use tf.ones here for compatibility and simplicity.
    return tf.ones((1, 5), dtype=tf.float32)

