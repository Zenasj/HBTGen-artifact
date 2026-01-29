# tf.random.uniform((B, 28, 28), dtype=tf.float32) ← inferred input shape from MNIST data input to the example model

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Reconstructing the MNIST classification model given in the issue
        # Input shape: (28, 28) grayscale images
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        self.dense1 = tf.keras.layers.Dense(512, activation=tf.nn.relu)
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    
    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        return self.dense2(x)

def my_model_function():
    # Return an instance of MyModel, no pretrained weights needed here
    return MyModel()

def GetInput():
    # Return a batch of random MNIST-like inputs with shape [batch_size, 28, 28]
    # Using a typical small batch (e.g., 32), float32 inputs normalized [0,1]
    batch_size = 32
    return tf.random.uniform((batch_size, 28, 28), minval=0, maxval=1, dtype=tf.float32)

