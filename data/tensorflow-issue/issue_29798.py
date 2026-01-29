# tf.random.uniform((B, 50, 50, 3), dtype=tf.float32)  # Assumed input shape and dtype from example in TPU code

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Build a convolutional model similar to the TPU example in chunks
        # This model uses 3 convolutional layers, batchnorm, ReLU, flatten, dense, batchnorm, and softmax
        self.conv_layers = []
        for _ in range(3):
            self.conv_layers.append(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='valid'))
            self.conv_layers.append(tf.keras.layers.BatchNormalization())
            self.conv_layers.append(tf.keras.layers.ReLU())
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(2)  # Output 2 classes as per example
        self.bn_after_dense = tf.keras.layers.BatchNormalization()
        self.softmax = tf.keras.layers.Softmax()

    def call(self, inputs, training=False):
        x = inputs
        for layer in self.conv_layers:
            # For BatchNorm layer pass training param
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                x = layer(x, training=training)
            else:
                x = layer(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.bn_after_dense(x, training=training)
        x = self.softmax(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor matching the input expected by MyModel
    # Based on example, input shape is [batch_size, 50, 50, 3]
    # Use batch size 32 arbitrarily
    batch_size = 32
    width = 50
    height = 50
    channels = 3
    # Use uniform floats between 0 and 1, float32 dtype
    return tf.random.uniform((batch_size, width, height, channels), dtype=tf.float32)

