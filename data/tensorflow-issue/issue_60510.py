# tf.random.uniform((B, 1), dtype=tf.float32) ← input shape inferred from model input shape (1,)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Shared Dense Encoder
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(2, activation='sigmoid')
        # Output layers extracting logits for two outputs with correct output shape (B, 1)
        # This is the fix: after slicing the lambda to get each output,
        # reshape to (B, 1) so accuracy metric works correctly.
        self.out1_lambda = tf.keras.layers.Lambda(lambda x: x[..., 0])
        self.out1_reshape = tf.keras.layers.Reshape((1,))
        self.out2_lambda = tf.keras.layers.Lambda(lambda x: x[..., 1])
        self.out2_reshape = tf.keras.layers.Reshape((1,))

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        out1 = self.out1_lambda(x)
        out1 = self.out1_reshape(out1)
        out2 = self.out2_lambda(x)
        out2 = self.out2_reshape(out2)
        return out1, out2

def my_model_function():
    # Instantiate the model and return.
    return MyModel()

def GetInput():
    # Generate a random input tensor with shape (batch_size=10, 1), dtype float32.
    # Matches the model input shape.
    return tf.random.uniform((10, 1), dtype=tf.float32)

# ---
# ### Explanation:
# - The original problem was the mismatch in output shapes from the Lambda layer slicing: the output shape was (batch_size,) instead of (batch_size, 1).  
# - This caused Keras' accuracy metric to behave incorrectly, treating outputs in a way that steps by batch count instead of sample count.
# - The solution that the user confirmed was to apply a `tf.keras.layers.Reshape((1,))` on the sliced outputs so the shape matches `(batch, 1)`.
# - This allows `accuracy` metric to compute correctly as fraction of total samples, consistent with expected behavior.
# - The input shape is `(batch_size, 1)` matching the example from the issue.
# - The two outputs are returned as a tuple `(out1, out2)` to matches the multiple output structure.
# - The model uses the same Dense layers as the original example.
# This satisfies all requirements:
# - Single class named `MyModel(tf.keras.Model)`.
# - `my_model_function()` returns an instance.
# - `GetInput()` returns a valid input tensor.
# - Includes fix insights in code comments.
# - Compatible with TF 2.20.0 XLA compilation (no unsupported operations).
# - No test or main blocks included.
# Let me know if you want me to also provide compilation example or metric usage!