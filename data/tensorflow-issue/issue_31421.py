# tf.random.uniform((B, 784), dtype=tf.float32) ← inferred input shape from MNIST flattened images

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Construct the same architecture as used in the example:
        # input shape: (784,)
        # Dense layers: 64 units relu, 64 units relu, 10 units softmax
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.dense3 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs):
        # Forward pass
        x = self.dense1(inputs)
        x = self.dense2(x)
        outputs = self.dense3(x)
        return outputs

def my_model_function():
    # Return a new instance of MyModel, weights not loaded here.
    # Training and weights loading can be done externally.
    return MyModel()

def GetInput():
    # Returns a random batch tensor of shape (B=8, 784), dtype float32, values in [0,1)
    # batch size chosen arbitrarily as 8
    BATCH_SIZE = 8
    return tf.random.uniform((BATCH_SIZE, 784), dtype=tf.float32)

