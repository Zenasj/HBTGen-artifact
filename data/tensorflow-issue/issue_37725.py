# tf.random.uniform((None, 28, 28, 1), dtype=tf.float32) ← Assumed input shape for Conv2D MNIST example

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Replicating the MNIST CNN architecture described in the issue
        self.conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1))
        self.conv2 = Conv2D(64, (3, 3), activation='relu')
        self.maxpool = MaxPooling2D(pool_size=(2, 2))
        self.dropout1 = Dropout(0.25)
        self.flatten = Flatten()
        self.dense1 = Dense(128, activation='relu')
        self.dropout2 = Dropout(0.5)
        self.dense2 = Dense(10, activation='softmax')

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.dropout1(x, training=training)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout2(x, training=training)
        return self.dense2(x)

def my_model_function():
    # Instantiate the model
    return MyModel()

def GetInput():
    # Return a batch of random MNIST-like grayscale images with shape [batch_size, 28, 28, 1]
    batch_size = 32  # Reasonable default batch size
    return tf.random.uniform((batch_size, 28, 28, 1), dtype=tf.float32)

