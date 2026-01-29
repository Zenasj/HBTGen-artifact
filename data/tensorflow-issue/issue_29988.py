# tf.random.uniform((BATCH_SIZE, 28, 28, 1), dtype=tf.float32) ← inferred input shape matching MNIST images batch

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple CNN architecture from the issue's model_fn
        self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1))
        self.pool = tf.keras.layers.MaxPooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input matching the MNIST input shape batch
    # Using BATCH_SIZE=64 as per original code, dtype float32
    BATCH_SIZE = 64
    # Input shape is (batch_size, 28, 28, 1)
    return tf.random.uniform((BATCH_SIZE, 28, 28, 1), dtype=tf.float32)

