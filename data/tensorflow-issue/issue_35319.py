# tf.random.uniform((1, 28, 28), dtype=tf.float32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        self.dense1 = tf.keras.layers.Dense(512, activation='relu')
        self.dense2 = tf.keras.layers.Dense(10, activation='softmax')
    
    def call(self, inputs):
        # inputs expected to be float32 normalized images (batch, 28, 28)
        x = self.flatten(inputs)
        x = self.dense1(x)
        return self.dense2(x)

def my_model_function():
    # Return a fresh instance of the model (untrained)
    return MyModel()

def GetInput():
    # Return a batch of 1 random MNIST-like float32 images scaled to [0, 1]
    # Shape: (1, 28, 28), dtype=tf.float32
    # This matches the input the model expects before quantization
    import numpy as np
    x = np.random.uniform(size=(1, 28, 28)).astype('float32')
    return tf.convert_to_tensor(x)

