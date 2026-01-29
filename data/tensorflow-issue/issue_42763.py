# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32) ← Using batch size B and MNIST input shape 28x28x1

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Gaussian noise with stddev roughly 0.75 as per original code
        self.noise = tf.keras.layers.GaussianNoise(stddev=0.75)
        self.flatten = tf.keras.layers.Flatten()
        # Dense layer with 10 units (num_classes), with ReLU activation as in original
        self.dense = tf.keras.layers.Dense(10, activation='relu')

    def call(self, x, training=False):
        # GaussianNoise is active only in training mode
        x = self.noise(x, training=training)
        x = self.flatten(x)
        x = self.dense(x)
        return x

def my_model_function():
    # Return an instance of MyModel; no pretrained weights
    return MyModel()

def GetInput():
    # Return a random tensor with shape (batch_size=110, 28, 28, 1)
    # float32 values normalized similarly as MNIST (0..1 float range)
    batch_size = 110
    return tf.random.uniform(shape=(batch_size, 28, 28, 1), minval=0, maxval=1, dtype=tf.float32)

