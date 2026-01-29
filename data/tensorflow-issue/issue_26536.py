# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32) ← typical input shape for this model example

import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__(name='MyModel')
        # Convolutional layers with BatchNorm and LeakyReLU activations
        self.conv1 = layers.Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same')
        self.conv1_bn = layers.BatchNormalization()
        self.conv1_act = layers.LeakyReLU()

        self.conv2 = layers.Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding='same')
        self.conv2_bn = layers.BatchNormalization()
        self.conv2_act = layers.LeakyReLU()

        self.conv3 = layers.Conv2D(256, kernel_size=(5, 5), strides=(2, 2), padding='same')
        self.conv3_bn = layers.BatchNormalization()
        self.conv3_act = layers.LeakyReLU()

        self.flatten = layers.Flatten()

        # Avoid naming conflict with Layer property 'output' by renaming to dense_output
        self.dense_output = layers.Dense(1)

    def call(self, inputs, training=True):
        x = self.conv1(inputs)
        x = self.conv1_bn(x, training=training)
        x = self.conv1_act(x)

        x = self.conv2(x)
        x = self.conv2_bn(x, training=training)
        x = self.conv2_act(x)

        x = self.conv3(x)
        x = self.conv3_bn(x, training=training)
        x = self.conv3_act(x)

        x = self.flatten(x)

        output = self.dense_output(x)

        return output


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Return a random tensor input matching the model expected shape
    # Assuming batch size 32, height 28, width 28, and 1 channel (grayscale)
    return tf.random.uniform((32, 28, 28, 1), dtype=tf.float32)

