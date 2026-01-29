# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32) ← inferred input shape for MNIST images preprocessed by scale()

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # This is the same architecture as the reported Sequential model
        self.conv2d = tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1))
        self.maxpool = tf.keras.layers.MaxPooling2D()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')

    def call(self, inputs, training=False):
        x = self.conv2d(inputs)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random input tensor shaped (batch_size, 28, 28, 1), normalized to [0,1]
    # Assumption: batch size 256 as used in batch from original code example
    batch_size = 256
    # Simulating 'scale' preprocessing by generating float32 in [0, 1]
    return tf.random.uniform((batch_size, 28, 28, 1), minval=0., maxval=1., dtype=tf.float32)

