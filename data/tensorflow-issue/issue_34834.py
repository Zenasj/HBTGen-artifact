# tf.random.uniform((B, 5, 5, 1), dtype=tf.float32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Define three Dense layers, no input_shape here; will handle in build
        self.dense0 = tf.keras.layers.Dense(10, name='dense0')
        self.dense1 = tf.keras.layers.Dense(10, name='dense1')
        self.dense2 = tf.keras.layers.Dense(10, name='dense2')

    def build(self, input_shape):
        # Build the network structure explicitly on dummy Inputs so Keras
        # can track inbound nodes properly for all layers, including dense2.
        # input_shape includes batch dimension as None
        # input_shape example: (None, 5, 5, 1)
        x = tf.keras.layers.Input(shape=input_shape[1:], name="input_x")
        x = self.dense0(x)
        x = tf.keras.layers.ReLU()(x)  # Use ReLU layer to keep node info
        x = self.dense1(x)
        # No activation function as a layer here, so following the original problem scenario:
        # Use tf.keras.activations.get('relu') which returns a function, not a layer.
        relu_fn = tf.keras.activations.get('relu')
        x = relu_fn(x)  # this breaks inbound nodes if done directly in call
        x = self.dense2(x)

    def call(self, x):
        x = self.dense0(x)
        # Use the ReLU layer - this sets inbound nodes correctly
        x = tf.keras.layers.ReLU()(x)
        x = self.dense1(x)
        # Show inbound nodes for dense1 - should be populated
        # Note: during graph tracing, print can be executed once
        tf.print("dense1 inbound_nodes shape:", len(self.dense1.inbound_nodes))

        # Now use relu activation function (not layer) - this will cause inbound node issues with next layer
        relu_fn = tf.keras.activations.get('relu')
        x = relu_fn(x)

        # dense2 after relu_fn; inbound nodes may be missing due to using activation function inline
        tf.print("dense2 inbound_nodes shape:", len(self.dense2.inbound_nodes))

        x = self.dense2(x)
        return x


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Return a random tensor with shape compatible with MyModel input
    # From code and comments: shape (5, 5, 1) per example, batch dimension arbitrary
    # Let's pick batch size 4 for example
    return tf.random.uniform((4, 5, 5, 1), dtype=tf.float32)

