# tf.random.uniform((B, 10), dtype=tf.float32) ← The input shape is (batch_size, 10)

import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Sequential is implemented as a list of layers applied one after another
        # Block has two Dense(10) layers in sequence
        # Base has two Blocks sequentially applied
        # Network wraps Base and returns its output

        # Define Block as a submodule
        class Block(tf.keras.layers.Layer):
            def __init__(self):
                super(Block, self).__init__()
                self.module = tf.keras.Sequential([
                    layers.Dense(10, name="dense"),
                    layers.Dense(10, name="dense_1"),
                ])

            def call(self, inputs):
                return self.module(inputs)

        # Define Base as a submodule that applies two Blocks sequentially,
        # where call returns a tuple (x, y), x = module(input), y = module(x)
        class Base(tf.keras.layers.Layer):
            def __init__(self):
                super(Base, self).__init__()
                self.module = tf.keras.Sequential([
                    Block(),
                    Block(),
                ])

            def call(self, inputs):
                x = self.module(inputs)
                y = self.module(x)
                return x, y

        # Define Network as submodule with child Base
        class Network(tf.keras.layers.Layer):
            def __init__(self):
                super(Network, self).__init__()
                self.child = Base()

            def call(self, inputs):
                return self.child(inputs)

        self.network = Network()

    def call(self, inputs):
        return self.network(inputs)


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Generate random input tensor with shape (batch_size, 10)
    # Using batch size 4 as an example
    return tf.random.uniform((4, 10), dtype=tf.float32)

