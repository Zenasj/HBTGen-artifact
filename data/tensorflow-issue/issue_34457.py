# tf.random.uniform((32, 32), dtype=tf.float32) ← Input shape inferred from model input_shape=(32,)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Define the layers mimicking the example Sequential model from the issue
        self.dense1 = tf.keras.layers.Dense(64, activation='relu', input_shape=(32,))
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.dense3 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.dense3(x)


def my_model_function():
    # Return an instance of MyModel; no pre-trained weights, initialized randomly as in example
    return MyModel()

def GetInput():
    # Return a random input tensor matching model input shape: (batch_size=32, features=32)
    # Batch size 32 is consistent with example batch_size in dataset generator
    return tf.random.uniform((32, 32), dtype=tf.float32)

