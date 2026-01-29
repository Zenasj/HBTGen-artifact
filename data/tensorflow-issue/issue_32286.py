# tf.random.uniform((None, 28, 28), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Flatten input 28x28 images to vectors
        self.flatten = tf.keras.layers.Flatten()
        # Dense layer with 128 units and ReLU activation
        self.dense1 = tf.keras.layers.Dense(128, activation=tf.nn.relu)
        # Output layer with 10 units and softmax activation for classification
        self.dense2 = tf.keras.layers.Dense(10, activation=tf.nn.softmax)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        return self.dense2(x)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor simulating batch of 32 grayscale 28x28 images
    # Values between 0 and 1 like normalized image inputs
    return tf.random.uniform((32, 28, 28), minval=0, maxval=1, dtype=tf.float32)

