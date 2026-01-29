# tf.random.uniform((None, 4), dtype=tf.float32) ← Input shape inferred from saved model "foo": input_shape=4

import tensorflow as tf

# Since the issue was about stitching two sequential models:
# 1) model foo: input 4 -> Dense(16, relu) -> Dense(16, relu) -> Dense(3, sigmoid)
# 2) model bar: input 3 -> Dense(16, relu) -> Dense(16, relu) -> Dense(2, sigmoid)
#
# The user wants to stitch them so that foo's output (shape 3) becomes bar's input.
#
# To fuse them into a single tf.keras.Model MyModel, we:
# - create submodules FooModel and BarModel with the respective layers
# - in the forward pass: run foo_model, then run bar_model on foo's output
#
# We will assume relu and sigmoid activations as given.

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Foo model layers
        self.foo_layer_0 = tf.keras.layers.Dense(16, activation='relu', name="foo_layer_0")
        self.foo_layer_1 = tf.keras.layers.Dense(16, activation='relu', name="foo_layer_1")
        self.foo_layer_2 = tf.keras.layers.Dense(3, activation='sigmoid', name="foo_layer_2")  # output shape 3
        
        # Bar model layers
        self.bar_layer_0 = tf.keras.layers.Dense(16, activation='relu', name="bar_layer_0")
        self.bar_layer_1 = tf.keras.layers.Dense(16, activation='relu', name="bar_layer_1")
        self.bar_layer_2 = tf.keras.layers.Dense(2, activation='sigmoid', name="bar_layer_2")  # output shape 2

    def call(self, inputs, training=False):
        # inputs shape: (batch_size, 4)
        # Forward through foo model
        x = self.foo_layer_0(inputs)
        x = self.foo_layer_1(x)
        foo_output = self.foo_layer_2(x)  # shape (batch_size, 3)
        
        # Forward through bar model using foo's output as input
        y = self.bar_layer_0(foo_output)
        y = self.bar_layer_1(y)
        bar_output = self.bar_layer_2(y)  # shape (batch_size, 2)
        
        return bar_output


def my_model_function():
    # Return a new instance of MyModel
    return MyModel()


def GetInput():
    # Return a random tensor input matching foo's input shape: (batch_size, 4)
    # Use batch_size=1 for simplicity
    return tf.random.uniform((1, 4), dtype=tf.float32)

