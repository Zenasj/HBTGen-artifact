# tf.random.normal((B, 3)) ← Input shape inferred from the provided example where input shape=(None, 3)

import tensorflow as tf

class inner_block(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__(name="inner")
        # Dense layer with 2 units; kernel variables saved with problematic naming in TF 2.8 checkpoint
        self.layer = tf.keras.layers.Dense(2, name="inner_dense")
        # A variable that is expected to save with correct naming
        self.var = tf.Variable([1.0, 2.0], name="inner_var")

    def call(self, x):
        return self.layer(x) + self.var

class model(tf.keras.Model):
    def __init__(self):
        super().__init__(name="outer")
        # Nested inner_block model
        self.inner_m = inner_block()
        # Outer Dense layer
        self.dense = tf.keras.layers.Dense(2, name="outer_dense")

    def call(self, x):
        x = self.inner_m(x)
        x = self.dense(x)
        return x

class MyModel(tf.keras.Model):
    """
    Fused model that replicates the described nested keras model structure:
    - an inner custom Layer with a Dense layer + variable
    - an outer model combining inner_block and another Dense layer

    This matches the structure from the GitHub issue to allow reproduction of the variable naming behavior.
    """
    def __init__(self):
        super().__init__(name="outer")
        self.inner_m = inner_block()
        self.dense = tf.keras.layers.Dense(2, name="outer_dense")

    def call(self, x):
        x = self.inner_m(x)
        x = self.dense(x)
        return x

def my_model_function():
    # Build the model since some variables are created on build/call
    m = MyModel()
    # The original example used input_shape (None, 3)
    m.build((None, 3))
    return m

def GetInput():
    # Return a random tensor with batch size 1 and 3 features as seen in example
    return tf.random.normal((1, 3))

