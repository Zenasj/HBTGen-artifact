# tf.random.normal((10000, 50), dtype=tf.float32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.Dense_in = tf.keras.layers.Dense(
            units=50,
            activation=tf.nn.relu,
            use_bias=True
        )
        self.Dense_hidden = tf.keras.layers.Dense(
            units=20,
            activation=tf.nn.relu,
            use_bias=True
        )
        self.Dense_out = tf.keras.layers.Dense(
            units=1,
            activation=None,
            use_bias=True
        )

    def call(self, inputs, training=True):
        x = self.Dense_in(inputs)
        x = self.Dense_hidden(x)
        x = self.Dense_out(x)
        return x

def my_model_function():
    # Returns an instance of MyModel with default initialization
    return MyModel()

def GetInput():
    # Returns random input tensor compatible with MyModel
    # Input shape matches dummy_data from example: (10000, 50)
    return tf.random.normal((10000, 50), dtype=tf.float32)

