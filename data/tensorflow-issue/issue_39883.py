# tf.random.uniform((B, 1), dtype=tf.float32) ← inferred input shape (batch size B, feature dim 1)

import os
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Using a very simple bias-only model to replicate the example from the test
        self.bias_layer = tf.keras.layers.Dense(1, use_bias=True, kernel_initializer='zeros', bias_initializer='zeros')

    def call(self, inputs):
        # Forward pass through bias-only layer
        return self.bias_layer(inputs)

def my_model_function():
    # Return an instance of MyModel
    model = MyModel()
    # Compile model to mimic example with SGD optimizer and mae loss as in test
    optimizer = tf.keras.optimizers.SGD(0.1)
    model.compile(loss='mae', optimizer=optimizer, metrics=['mae'])
    return model

def GetInput():
    # Create random batch input with shape (B, 1) matching expected input shape for MyModel
    # Batch size 8 chosen to match batching in example test case
    batch_size = 8
    input_shape = (batch_size, 1)

    # Random float32 input tensor
    return tf.random.uniform(input_shape, dtype=tf.float32)

