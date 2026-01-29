# tf.random.uniform((B, 1), dtype=tf.float32) ← Inferred input shape: batch size B, feature dimension 1

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # A single Dense layer with 1 neuron and sigmoid activation,
        # initialized with weights and bias same as in the original example
        self.dense = tf.keras.layers.Dense(
            1,
            activation='sigmoid',
            name='L1',
            kernel_initializer=tf.constant_initializer([[2.0]]),
            bias_initializer=tf.constant_initializer([-4.5])
        )
        
    def call(self, inputs):
        # Forward pass through the logistic neuron layer
        return self.dense(inputs)

def my_model_function():
    # Return an instance of MyModel. Weights are set via initializers above.
    return MyModel()

def GetInput():
    # Generate a random batch of inputs compatible with MyModel:
    # shape: (batch_size, 1), dtype float32
    # batch size chosen arbitrarily (e.g., 6 to match original example size)
    batch_size = 6
    return tf.random.uniform((batch_size, 1), dtype=tf.float32)

