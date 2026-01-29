# tf.random.uniform((32, 28, 28, 1), dtype=tf.float32) ← Assumed batch size 32, MNIST images 28x28 grayscale

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # MNIST image shape 28x28x1 assumed from dataset
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28, 1))
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dropout(x, training=training)  # training flag used for dropout behavior consistency
        return self.dense2(x)

def my_model_function():
    # Return a fresh instance of the model
    return MyModel()

def GetInput():
    # Return a random tensor simulating a batch of MNIST grayscale images
    # Batch size 32, height 28, width 28, channels 1
    return tf.random.uniform((32, 28, 28, 1), dtype=tf.float32)

# Note:
# This model structure is based on the reported code to reproduce accuracy discrepancies between 
# eager and graph modes in TensorFlow 2.0 alpha.
# The dropout layer uses the training flag in call() to ensure consistent behavior during training.
# The input shape is inferred from MNIST dataset shape.

