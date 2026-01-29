# tf.random.uniform((B, 5), dtype=tf.float32) ← Input shape inferred from original example: input tensors have shape (batch_size, 5)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define a simple single Dense layer as in the original example
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs, training=False):
        # Forward pass through the Dense layer
        return self.dense(inputs)

def my_model_function():
    # Create and compile an instance of MyModel analogous to the original Sequential model
    model = MyModel()
    # Compile so the model has loss & optimizer (not strictly needed here for forward, but matches original example)
    model.compile(loss="mse", optimizer="sgd")
    return model

def GetInput():
    # Return a random tensor input matching the expected input shape (batch_size=10 arbitrarily chosen)
    return tf.random.uniform((10, 5), dtype=tf.float32)

