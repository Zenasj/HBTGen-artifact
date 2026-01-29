# tf.random.uniform((B, 20), dtype=tf.float32) ← Input shape inferred from issue example using shape (batch_size, 20)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define a simple model matching the example:
        # Input shape: (None, 20)
        # Single Dense layer with 10 units
        self.dense = tf.keras.layers.Dense(10)

    def call(self, inputs, training=False):
        # Forward pass through the dense layer
        return self.dense(inputs)

def my_model_function():
    # Create an instance of MyModel
    model = MyModel()
    # Explicitly build the model by passing an input shape batch - ensures the model is built before fit
    model.build(input_shape=(None, 20))
    # Compile with SGD optimizer and mse loss as seen in the example
    model.compile(optimizer=tf.keras.optimizers.SGD(), loss='mse')
    return model

def GetInput():
    # Return a random tensor with shape compatible with model input: (batch_size, 20)
    # Batch size is chosen arbitrarily as 5 to match example data shape (5, 20)
    return tf.random.uniform((5, 20), dtype=tf.float32)

