# tf.random.uniform((B, 1), dtype=tf.float32) ← Input shape inferred from example: single feature vectors of shape (batch_size, 1)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Construct a sequential-like stack of 200 Dense(1) layers.
        # This replicates the "extreme example" model with deep stacking of layers.
        self.dense_layers = [tf.keras.layers.Dense(1) for _ in range(200)]

    def call(self, inputs, training=False):
        x = inputs
        for layer in self.dense_layers:
            x = layer(x)
        return x


def my_model_function():
    # Returns an instance of MyModel.
    # The example compiles with loss="mse" and uses run_eagerly=True in the minimal repro.
    return MyModel()


def GetInput():
    # Return a random input tensor consistent with the model input shape: (batch_size, 1).
    # Batch size chosen as 1 for minimal test, matching example.
    return tf.random.uniform((1, 1), dtype=tf.float32)

