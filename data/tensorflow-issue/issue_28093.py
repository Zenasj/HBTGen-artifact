# tf.random.uniform((None, 28, 28), dtype=tf.float32) ← Inferred input shape from MNIST example (batch unspecified)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Model structure from the issue's MNIST example
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        self.dense1 = tf.keras.layers.Dense(512, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random batch of inputs matching (batch_size, 28, 28) for the MNIST example
    # Assume batch size 32 as a common default
    return tf.random.uniform((32, 28, 28), dtype=tf.float32)

