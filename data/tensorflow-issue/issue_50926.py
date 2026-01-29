# tf.random.uniform((B, 28, 28), dtype=tf.float32) ← inferred input shape from MNIST dataset flattened input

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Sequential model matching the example for MNIST classification
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10)  # logits output

    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense1(x)
        logits = self.dense2(x)
        return logits

def my_model_function():
    # Return an instance of MyModel
    model = MyModel()
    # Note: weights not preloaded; could be loaded if needed externally
    return model

def GetInput():
    # MNIST input shape is (batch_size, 28, 28), grayscale images normalized to [0, 1]
    # Use batch size 128 (same as original script)
    batch_size = 128
    # Create random float tensor in [0,1) to simulate normalized MNIST images
    return tf.random.uniform((batch_size, 28, 28), dtype=tf.float32)

