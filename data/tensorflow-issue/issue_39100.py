# tf.random.uniform((B, 2), dtype=tf.float32)  ← Input shape is (batch_size, 2) as per original model input

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Reconstruct the model described: input=2 dims, Dense(8, tanh), Dense(4, sigmoid), Dense(1, sigmoid)
        self.dense1 = tf.keras.layers.Dense(8, activation='tanh', name='layer1')
        self.dense2 = tf.keras.layers.Dense(4, activation='sigmoid', name='layer2')
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid', name='outputlayer')

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.output_layer(x)

def my_model_function():
    # Return a fresh instance of MyModel
    return MyModel()

def GetInput():
    # Return a random input tensor with shape (batch_size=1, 2) matching model input size.
    # Using uniform distribution in range [0, 1)
    return tf.random.uniform((1, 2), dtype=tf.float32)

