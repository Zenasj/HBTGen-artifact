# tf.random.uniform((B, 784), dtype=tf.float32)

import tensorflow as tf
import numpy as np

class BuggyRandom(tf.keras.layers.Layer):
    # Reproduces the old "buggy" behavior where tf.random.uniform inside tf.function
    # and certain control flows produces same random number sequences across runs.
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # A boolean variable to control printing randomness inside call
        self.boolean_var = tf.Variable(True)
    def call(self, inputs):
        if self.boolean_var:
            # This triggers the problematic usage scenario:
            # tf.random.uniform inside a tf.function with conditional variable,
            # causing deterministic/repeated sequences unexpectedly.
            tf.print(tf.random.uniform([5,]), summarize=-1)
        return inputs

class FixedRandom(tf.keras.layers.Layer):
    # Uses the recommended tf.random.Generator API to produce truly different
    # random numbers across different runs even inside tf.function.
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize generator with a random seed each instantiation to simulate true randomness
        self.g = tf.random.Generator.from_seed(np.random.randint(2147483647))
    def call(self, inputs):
        # Generate and print random uniform numbers via the Generator instance
        tf.print(self.g.uniform([5,]), summarize=-1)
        return inputs

class MyModel(tf.keras.Model):
    """
    Combines both buggy and fixed random layers, encapsulating the old and new APIs.

    The call() method will run a stack of buggy layers followed by fixed layers, showing
    the deterministic behavior of buggy and the improved randomness of fixed approach.

    Output: A dict with:
      'buggy_outputs': output tensor after buggy layers,
      'fixed_outputs': output tensor after fixed layers,
      'random_diff': numeric tensor measuring absolute difference of last buggy vs last fixed printouts,
                     simulated here as the difference of last random numbers generated for illustration.
    """

    def __init__(self):
        super().__init__()
        # 4 buggy random layers as in original examples
        self.buggy1 = BuggyRandom()
        self.buggy2 = BuggyRandom()
        self.buggy3 = BuggyRandom()
        self.buggy4 = BuggyRandom()

        # Assign some boolean_vars to False to demo partial printing (like in example)
        # Here we do it after build in call to avoid tf.Variable assignment error on init

        # 4 fixed random layers as in final example
        self.fixed1 = FixedRandom()
        self.fixed2 = FixedRandom()
        self.fixed3 = FixedRandom()
        self.fixed4 = FixedRandom()

    def call(self, inputs, training=None):
        # Assign boolean_vars after first run (to mimic example behavior)
        # In real scenario you'd want these consistent and not change in call,
        # but we do it here to simulate example setup:
        self.buggy2.boolean_var.assign(False)
        self.buggy4.boolean_var.assign(False)

        # Pass input through buggy layers
        x = self.buggy1(inputs)
        x = self.buggy2(x)
        x = self.buggy3(x)
        buggy_out = self.buggy4(x)

        # Pass same input through fixed random layers
        y = self.fixed1(inputs)
        y = self.fixed2(y)
        y = self.fixed3(y)
        fixed_out = self.fixed4(y)

        # For demonstration, compute a numeric difference between last buggy and fixed layers'
        # random numbers is not accessible directly (tf.print outputs only).
        # Instead, we illustrate by generating two fresh random uniform tensors and taking diff.
        # This is a placeholder for "comparison" logic.
        diff_tensor = tf.abs(self.fixed4.g.uniform([5,]) - tf.random.uniform([5,]))

        return {
            'buggy_outputs': buggy_out,
            'fixed_outputs': fixed_out,
            'random_diff': diff_tensor
        }

def my_model_function():
    # Return an instance of MyModel, ready to run
    return MyModel()

def GetInput():
    # Return a random input tensor matching the input expected by MyModel: shape (batch_size, 784)
    # Using batch size 64 as in examples
    return tf.random.uniform((64, 784), dtype=tf.float32)

