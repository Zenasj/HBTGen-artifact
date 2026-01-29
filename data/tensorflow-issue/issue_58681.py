# tf.random.uniform((B, 28, 28), dtype=tf.float32) ← inferred from the MNIST input shape in the example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # This model replicates the MNIST example used in the issue.
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense1(x)
        if training:
            x = self.dropout(x, training=training)
        return self.dense2(x)

def my_model_function():
    # Return an instance of MyModel, no special initializations beyond default
    return MyModel()

def GetInput():
    # Return a random input tensor shaped like an MNIST batch example (batch size arbitrarily chosen as 32)
    # dtype tf.float32 matches the expected input dtype for model layers
    batch_size = 32
    return tf.random.uniform((batch_size, 28, 28), dtype=tf.float32)

