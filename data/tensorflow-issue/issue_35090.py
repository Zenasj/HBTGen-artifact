# tf.random.uniform((B=1,), dtype=tf.float32) ← This matches random.uniform([1], 0.2, 5.0) calls in the original examples

import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    This model encapsulates two separate random.uniform calls, simulating
    the Dataset.map scenario where two different map calls with the same seed
    produce the same random sequences.

    It compares the outputs of the two random calls seeded identically
    and returns a boolean tensor indicating if they are equal (same random numbers).

    This mimics the core of the issue described:
    - Two calls with the same seed produce identical random outputs.
    """
    def __init__(self, seed=None):
        super().__init__()
        # Seed here simulates global/graph-level seed that causes identical sequences in different map calls.
        self.seed = seed

    @tf.function(jit_compile=True)
    def call(self, inputs=None):
        # inputs is ignored; random ops do not depend on inputs in the original issue
        
        # Generate first random uniform scalar with given seed
        rand1 = tf.random.uniform([1], 0.2, 5.0, seed=self.seed)
        # Generate second random uniform scalar with the same seed
        rand2 = tf.random.uniform([1], 0.2, 5.0, seed=self.seed)

        # Compare if both generated values are exactly equal (elementwise)
        is_equal = tf.reduce_all(tf.equal(rand1, rand2))
        # Output a dictionary for clarity: random1, random2, and comparison boolean
        return {"rand1": rand1, "rand2": rand2, "equal": is_equal}

def my_model_function():
    # Here we instantiate the model with a fixed seed to simulate the behavior that causes repeated sequences.
    # The seed aligns with the original issue seed (e.g. 88).
    return MyModel(seed=88)

def GetInput():
    # The original random.uniform call does not actually use the dataset element as input,
    # so we return a dummy tensor of shape () (scalar) to accommodate the call.
    #
    # This matches the tf.data.Dataset.range(0,4) elements passed, but unused.
    return tf.constant(0, dtype=tf.int64)

