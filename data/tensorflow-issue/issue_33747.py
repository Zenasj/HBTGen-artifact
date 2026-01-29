# tf.random.uniform((B, 28, 28, 1), dtype=tf.float32) ← Input shape inferred from fashion_mnist dataset with a single grayscale channel

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Model layers similar to fashion_mnist example in issue
        self.conv = tf.keras.layers.Conv2D(128, (3, 3), activation=None)
        self.pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
        self.elu = tf.keras.layers.Activation('elu')
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(10)
        self.softmax = tf.keras.layers.Activation('softmax')

    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.pool(x)
        x = self.elu(x)
        x = self.flatten(x)
        x = self.dense(x)
        return self.softmax(x)

def my_model_function():
    # Return an instance of the model, no pretrained weights assumed
    return MyModel()

def GetInput():
    # Batch size chosen arbitrarily as 32 for example usage
    batch_size = 32
    height, width, channels = 28, 28, 1
    # Return float32 tensor with values in [0, 1), matching dataset and model input
    return tf.random.uniform((batch_size, height, width, channels), dtype=tf.float32)

