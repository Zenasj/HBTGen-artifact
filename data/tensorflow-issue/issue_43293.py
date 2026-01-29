# tf.random.uniform((B, None, 1), dtype=tf.float32) ← Input shape inferred from tf.keras.Input(shape=(None, 1))

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Two Dense layers as in the original example
        self.dense1 = tf.keras.layers.Dense(1)
        self.dense2 = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        output = self.dense2(x)
        # Return both intermediate and final outputs as a tuple for usage in loss and accuracy
        # This matches the usage of intermediate layers in the original code (x and output)
        return x, output

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input matching the model's expected input shape
    # Input shape is (B, T, 1) with variable T (None), we pick T=10 arbitrarily for testing
    B = 128
    T = 10
    return tf.random.uniform((B, T, 1), dtype=tf.float32)

# Additional helper functions to compute loss and accuracy to reflect the logic in the original issue:

@tf.function
def compute_loss(model, inputs, targets):
    """
    Computes loss using intermediate layer x and final output.
    error = output - targets + x_slice (x[:, 0])
    Mean squared error is returned.
    """
    x, outputs = model(inputs)
    output = outputs[:, 0, 0]      # first output's first element along last dims (B, T, 1) → (B,)
    x_slice = x[:, 0, 0]           # intermediate layer's first slice along time axis, shape (B,)
    error = output - tf.squeeze(targets, axis=-1) + x_slice
    return tf.reduce_mean(tf.square(error))


@tf.function
def compute_hits(model, inputs, targets):
    """
    Computes the count of "hits" where:
    (output - target + x_slice) < 0.01
    Returns total count of hits as an integer tensor.
    """
    x, outputs = model(inputs)
    output = outputs[:, 0, 0]
    x_slice = x[:, 0, 0]
    condition = output - tf.squeeze(targets, axis=-1) + x_slice < 0.01
    hits = tf.reduce_sum(tf.where(condition, 1, 0))
    return hits

