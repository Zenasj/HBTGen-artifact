# tf.random.uniform((64, 512, 512, 12), dtype=tf.float32)  <- inferred input shape from example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Replicating the toy model from the issue with Conv2D layers and activations
        self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')
        self.conv2 = tf.keras.layers.Conv2D(64, 1, activation='relu', padding='same')
        self.conv3 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')
        self.conv4 = tf.keras.layers.Conv2D(1, 1, padding='same')
        self.activation = tf.keras.layers.Activation('linear')

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.activation(x)
        return x


def my_model_function():
    # Return an instance of MyModel, with no pretrained weights assumed
    return MyModel()


def GetInput():
    # Returns a tf.Tensor shape (64, 512, 512, 12) matching example batch size and channels
    # Using float32 dtype as typical for conv models
    return tf.random.uniform((64, 512, 512, 12), dtype=tf.float32)

