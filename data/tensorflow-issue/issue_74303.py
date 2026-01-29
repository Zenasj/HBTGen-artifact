# tf.random.uniform((128, 28, 28, 1), dtype=tf.float32) ← inferred from MNIST input shape and batch_size in the issue

import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Reproduce the MNIST convnet structure as given in the issue's example
        self.conv1 = layers.Conv2D(32, kernel_size=(3, 3), activation="relu")
        self.pool1 = layers.MaxPooling2D(pool_size=(2, 2))
        self.conv2 = layers.Conv2D(64, kernel_size=(3, 3), activation="relu")
        self.pool2 = layers.MaxPooling2D(pool_size=(2, 2))
        self.flatten = layers.Flatten()
        self.dropout = layers.Dropout(0.5)
        self.dense = layers.Dense(10, activation="softmax")

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dropout(x, training=training)
        return self.dense(x)

def my_model_function():
    # Return an instance of MyModel as per the original Keras Sequential configuration
    return MyModel()

def GetInput():
    # Generate a random batch input tensor consistent with MNIST shape (batch, 28, 28, 1)
    # batch size is chosen as 128 like in the issue's example
    batch_size = 128
    input_shape = (28, 28, 1)
    return tf.random.uniform((batch_size,) + input_shape, dtype=tf.float32)

