# tf.random.uniform((10, 215, 215, 1), dtype=tf.float32) ← inferred input shape matching training data

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Following the original model architecture from the issue
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5), strides=(2, 2))
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), strides=(2, 2))
        self.conv3 = tf.keras.layers.Conv2D(filters=128, kernel_size=(5, 5), strides=(2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=128)
        self.batchnorm = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        # Adjust output layer to one unit with sigmoid activation following issue comments for correct metric behavior
        self.output_layer = tf.keras.layers.Dense(units=1, activation='sigmoid')

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.batchnorm(x, training=training)
        x = self.relu(x)
        output = self.output_layer(x)
        return output

def my_model_function():
    """
    Returns an instance of MyModel.
    """
    return MyModel()

def GetInput():
    """
    Returns a random tensor input matching the expected input of MyModel:
    shape (10, 215, 215, 1) with dtype float32, values in typical range.
    """
    # Using uniform distribution 0-1 as typical image float input
    return tf.random.uniform(shape=(10, 215, 215, 1), dtype=tf.float32)

