# tf.random.uniform((B, 500000), dtype=tf.float32) ← input shape is (batch_size, input_dim=500000)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define a simple feedforward network as described in the issue:
        # Input dimension = 500000
        self.dense1 = tf.keras.layers.Dense(64, activation='relu', input_shape=(500000,))
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.output_layer = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=False):
        # Forward pass
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.output_layer(x)

def my_model_function():
    # Return an instance of MyModel with no pretrained weights
    return MyModel()

def GetInput():
    # Return a random tenosr that matches input expected by MyModel
    # Use float32 dtype matching typical TF default and realistic input data
    batch_size = 32  # batch size consistent with example in issue
    input_dim = 500000
    # Random uniform input with shape (batch_size, input_dim)
    return tf.random.uniform(shape=(batch_size, input_dim), dtype=tf.float32)

