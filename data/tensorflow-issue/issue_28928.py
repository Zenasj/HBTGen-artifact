# tf.random.uniform((B, 2), dtype=tf.float32) ← Input shape inferred from example: 2 features per sample

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Single Dense layer as per example
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs):
        # inputs can be a tensor or a dictionary with key "input"
        # Support either form for robustness based on the issue reproducibility context
        if isinstance(inputs, dict):
            x = inputs.get("input", None)
            if x is None:
                raise ValueError("Expected input dict to have key 'input'")
        else:
            x = inputs
        return self.dense(x)

def my_model_function():
    # Instantiate and return the model
    return MyModel()

def GetInput():
    # Generate a random tensor matching the input shape (batch size, 2 features)
    # Batch size picked arbitrarily as 10 to match example batch_size
    return tf.random.uniform((10, 2), dtype=tf.float32)

