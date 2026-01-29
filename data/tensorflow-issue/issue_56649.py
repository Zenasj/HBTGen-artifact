# tf.random.uniform((B, 28, 28), dtype=tf.float32) ← inferred input shape for MNIST example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Flatten layer to convert 28x28 images to 784 vector
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        # Dense layer with 128 units and ReLU activation
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        # Dropout layer with 0.2 dropout rate
        self.dropout = tf.keras.layers.Dropout(0.2)
        # Output Dense layer with 10 units (for 10 classes) and softmax activation
        self.output_layer = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        return self.output_layer(x)

def my_model_function():
    # Return an instance of MyModel, which matches the original MNIST example model used in issue
    return MyModel()

def GetInput():
    # Return a random tensor matching the MNIST image shape used as input
    # Batch size of 32 chosen arbitrarily for example inputs
    return tf.random.uniform((32, 28, 28), dtype=tf.float32)

