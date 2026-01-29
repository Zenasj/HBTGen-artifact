# tf.random.uniform((B, 28, 28), dtype=tf.float32) ← inferred input shape from MNIST dataset in the issue

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.batchnorm = tf.keras.layers.BatchNormalization(renorm=False)
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(10)
        
    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.batchnorm(x, training=training)
        x = self.dropout(x, training=training)
        logits = self.dense2(x)
        return logits

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a batch of random MNIST-like inputs (28x28 grayscale images)
    # Assuming batch size of 32 for example
    batch_size = 32
    # Normalized input as the original code does x_train, x_test = x / 255.0
    # Here we generate random floats between 0 and 1 to simulate normalized inputs
    return tf.random.uniform((batch_size, 28, 28), minval=0, maxval=1, dtype=tf.float32)

