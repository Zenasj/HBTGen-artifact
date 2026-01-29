# tf.random.normal((1, 64), dtype=tf.float32) ← Input shape inferred from the example (batch=1, features=64)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Construct two Dense layers with the same size (64 units each)
        # This replicates the original problem setup where having two dense layers of same size caused allocation error in TFLite EdgeTPU quantization
        self.dense1 = tf.keras.layers.Dense(64)
        self.dense2 = tf.keras.layers.Dense(64)

    @tf.function
    def call(self, inputs):
        # Forward pass through two dense layers in sequence
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x

def my_model_function():
    # Initialize the model, weights are randomly initialized but should suffice as placeholder
    model = MyModel()
    # Build the model by calling once with input to create variables (useful for tf.function tracing or compilation)
    dummy_input = GetInput()
    _ = model(dummy_input)
    return model

def GetInput():
    # Return a random tensor matching the expected input shape (batch size 1, feature size 64)
    # Use tf.random.normal to match the example's representative dataset generation using tf.random.normal.
    return tf.random.normal(shape=(1, 64), dtype=tf.float32)

