# tf.random.uniform((B, 30), dtype=tf.float32) ← Input shape inferred from code example with 30 features

import tensorflow as tf
import numpy as np

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # This model replicates the architecture described in the issue's Sequential model:
        # 4 Dense layers with 100 units and ReLU activation, followed by a Dense output layer with linear activation.
        self.dense1 = tf.keras.layers.Dense(100, activation='relu')
        self.dense2 = tf.keras.layers.Dense(100, activation='relu')
        self.dense3 = tf.keras.layers.Dense(100, activation='relu')
        self.dense4 = tf.keras.layers.Dense(100, activation='relu')
        self.output_layer = tf.keras.layers.Dense(30, activation='linear')  # Output shape = input shape

    @tf.function(jit_compile=True)
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        return self.output_layer(x)

def my_model_function():
    # Instantiate and return the model
    return MyModel()

def GetInput():
    # Generate random input matching the expected shape [batch_size, 30]
    # The examples use batches of 10^5 in benchmarking, but for demonstration we use a moderate batch size
    batch_size = 64
    # Use float32 uniform random inputs here to simulate typical input data like the normal distribution in example
    # In the original snippet, linear inputs are generated with np.random.randn (normal dist), but uniform is fine as placeholder
    return tf.random.uniform((batch_size, 30), dtype=tf.float32)

