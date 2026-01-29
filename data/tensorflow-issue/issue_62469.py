# tf.random.uniform((1, 10, 10, 3), dtype=tf.float32) ← Input shape and dtype inferred from model Input layer

import tensorflow as tf

# Custom fused operation using tf.function with experimental_implements signature
def get_implements_signature():
    # Signature marking this function as a fusable op for TFLite converter
    implements_signature = [
        'name: "exp_sin"',
        'attr {key: "tfl_fusable_op" value { b: true } }',
    ]
    return " ".join(implements_signature)


@tf.function(experimental_implements=get_implements_signature())
def exp_sin(x):
    # Computes exp followed by sin: sin(exp(x))
    x = tf.math.exp(x)
    x = tf.math.sin(x)
    return x


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Custom layer wrapping the fused exp_sin op
        self.custom_example = CustomExample()
        # Conv2D layer as in the example code
        self.conv = tf.keras.layers.Conv2D(filters=8, kernel_size=4)
    
    def call(self, inputs):
        # Pipe input through custom fused op layer, then conv2d
        x = self.custom_example(inputs)
        x = self.conv(x)
        return x


class CustomExample(tf.keras.layers.Layer):
    def call(self, inputs):
        # Call the fused op function exp_sin
        return exp_sin(inputs)


def my_model_function():
    # Return an instance of MyModel initialized
    return MyModel()


def GetInput():
    # Generate a random float32 tensor matching input shape (batch=1, H=10,W=10,C=3)
    return tf.random.uniform((1, 10, 10, 3), dtype=tf.float32)

