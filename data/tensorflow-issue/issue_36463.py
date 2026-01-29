# tf.random.uniform((32, 784), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Create a Sequential model with two Dense layers as described
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(units=10, name="d1"),
            tf.keras.layers.Dense(units=20, name="d2"),
        ])

        # Also create two Dense layers independently to compare behavior with Sequential
        self.dense1 = tf.keras.layers.Dense(units=10, name="d1")
        self.dense2 = tf.keras.layers.Dense(units=20, name="d2")

    def build(self, input_shape):
        # Build the Sequential and the individual Dense layers with the same input_shape
        # We do this manually to simulate the name_scope behavior described.
        # Note: The issue described is that Sequential's variable names ignore surrounding
        # tf.name_scope in eager mode but Dense respects it.
        # Here, no explicit tf.name_scope is applied in build; actual name_scope usage
        # happens outside before build.

        self.seq.build(input_shape)
        self.dense1.build(input_shape)
        self.dense2.build(input_shape)
        super().build(input_shape)

    def call(self, inputs):
        # Forward pass simply returns outputs of the Sequential model,
        # but also return outputs of the 2 Dense layers individually to allow comparison.
        seq_out = self.seq(inputs)
        dense_out1 = self.dense1(inputs)
        dense_out2 = self.dense2(dense_out1)

        # For demonstration, return a dict comparing output sums - to "compare" the models.
        # In real usage, output can be customized. Here including diff metric as example.
        diff1 = tf.reduce_sum(tf.abs(seq_out - dense_out2))

        # Return a tuple with all outputs and diff between the models for inspection.
        return {
            "seq_output": seq_out,
            "dense_sequential_output": dense_out2,
            "diff": diff1
        }

def my_model_function():
    # Instantiate MyModel normally. Users can encapsulate any name_scope usage outside.
    return MyModel()

def GetInput():
    # Input shape should be compatible with input_shape=[32, 784] used in build calls above.
    # The expected shape is (batch_size=32, features=784), dtype float32 for Dense layers
    return tf.random.uniform((32, 784), dtype=tf.float32)

