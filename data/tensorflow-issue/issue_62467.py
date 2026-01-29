# tf.random.uniform((17, 64, 59, 1, 1), dtype=tf.float32) ← inferred input shape from issue's comments and code

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Instantiate Model1 and Model2 as submodules inside MyModel
        self.model1 = self.Model1()
        self.model2 = self.Model2()

    def call(self, inp):
        # Run both models on the same input
        out1 = self.model1(inp)
        out2 = self.model2(inp)

        # Perform a numeric comparison between their outputs.
        # Use relative tolerance of 0.01 as implied from the issue.
        # Return a boolean tensor indicating if outputs are close within the tolerance.
        # Also output the absolute difference tensor for detailed insight.
        are_close = tf.reduce_all(tf.math.abs(out1 - out2) <= 0.01 * tf.math.abs(out2))
        abs_diff = tf.math.abs(out1 - out2)
        # Return a tuple: (comparison boolean scalar, absolute difference tensor)
        return are_close, abs_diff

    class Model1(tf.keras.Model):
        def __init__(self):
            super().__init__()

        @tf.function(jit_compile=True)
        def __call__(self, inp):
            # Implements (abs(inp) + inp) * (-inp)
            v0_0 = tf.abs(inp)
            v2_0 = tf.negative(inp)
            v4_0 = tf.add(v0_0, inp)
            v5_0 = tf.multiply(v2_0, v4_0)
            return v5_0

    class Model2(tf.keras.Model):
        def __init__(self):
            super().__init__()

        @tf.function(jit_compile=True)
        def __call__(self, inp):
            # Implements -inp * abs(inp) + inp * -inp
            v2_0 = tf.negative(inp)
            v3_0 = tf.abs(inp)
            v4_0 = tf.multiply(v3_0, v2_0)
            v5_0 = tf.multiply(inp, v2_0)
            v6_0 = tf.add(v4_0, v5_0)
            return v6_0

def my_model_function():
    # Return an instance of MyModel with embedded Model1 and Model2 submodules.
    return MyModel()

def GetInput():
    # Return a random tensor input consistent with the input shape and dtype used in the issue.
    # Shape: [17, 64, 59, 1, 1], dtype: tf.float32
    # Random uniform values in range [-1, 1] to span positive and negative inputs.
    return tf.random.uniform(shape=(17, 64, 59, 1, 1), minval=-1.0, maxval=1.0, dtype=tf.float32)

