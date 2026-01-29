# tf.random.uniform((B, 28*28), dtype=tf.float32) ← Input shape inferred from MNIST flattened images

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define layers as per the original Keras model architecture
        self.dense1 = tf.keras.layers.Dense(300, activation="relu", input_shape=(28*28,))
        self.dense2 = tf.keras.layers.Dense(100, activation="relu")
        self.dense3 = tf.keras.layers.Dense(10, activation="softmax")

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.dense3(x)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random input tensor with shape (batch_size=32, 28*28) matching MyModel expected input
    # Use float32 values in [0,1) consistent with MNIST preprocessing
    batch_size = 32
    return tf.random.uniform((batch_size, 28*28), minval=0, maxval=1, dtype=tf.float32)

