# tf.random.uniform((B, 28, 28), dtype=tf.float32) ← inferred input shape based on Flatten input_shape=(28,28)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define a simple model matching the example from the issue
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        return self.dense2(x)

def my_model_function():
    # Instantiate and return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate a random float32 tensor with batch size 1 to match input (28,28)
    # This matches the expected input shape by the model (batch, 28, 28)
    return tf.random.uniform((1, 28, 28), dtype=tf.float32)

