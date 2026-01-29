# tf.random.uniform((B, 50), dtype=tf.float32)  ← Assumed input shape (batch_size, 50) based on input_shape=[100, 50] in example

import tensorflow as tf

class CustomLayer(tf.keras.layers.Layer):
    def __init__(self, units_base, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.units_base = units_base
        # We defer building the Dense layer until build time
        self.layer = None

    def build(self, input_shape):
        # input_shape is a TensorShape, get last dim as int:
        input_dim = input_shape[-1] if isinstance(input_shape, (list, tuple)) else input_shape[-1]
        # units_base + input_dim for output units of Dense
        self.layer = tf.keras.layers.Dense(self.units_base + input_dim)
        # Mark layer built
        self.built = True

    def call(self, inputs, *args, **kwargs):
        # Call the Dense layer on inputs
        return self.layer(inputs)


class MyModel(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # We defer creation of CustomLayers until build, because we need input shape

        self.l0 = None
        self.l1 = None

    def build(self, input_shape):
        # input_shape: TensorShape or list of ints (batch_size, 50)
        # For units_base, use input feature dimension (last dim)
        if hasattr(input_shape, '__len__'):
            input_dim = input_shape[-1]
        else:
            input_dim = input_shape[-1]

        # Initialize two CustomLayer instances using the input feature size
        self.l0 = CustomLayer(units_base=input_dim)
        self.l1 = CustomLayer(units_base=input_dim)
        # Build layers explicitly with input shape excluding batch (batch may be None)
        self.l0.build(input_shape)
        intermediate_shape = tf.TensorShape([input_shape[0], self.l0.units_base + input_dim])
        self.l1.build(intermediate_shape)

        self.built = True

    def call(self, inputs, *args, **kwargs):
        x = self.l0(inputs)
        x = self.l1(x)
        return x


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    # Based on example input_shape=[100, 50], batch size of 100 and feature size of 50
    batch_size = 100
    feature_dim = 50
    return tf.random.uniform((batch_size, feature_dim), dtype=tf.float32)

