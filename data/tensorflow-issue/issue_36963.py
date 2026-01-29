# tf.random.uniform((B, H, W, C), dtype=...) ← Add a comment line at the top with the inferred input shape
# Based on the test script inputs shapes used: (1,100,100,20) and (1,80,80,20) as example inputs

import tensorflow as tf
keras = tf.keras

class Scale(keras.layers.Layer):
    def __init__(self, destination_channel=None, name="Scale", **kwargs):
        super().__init__(name=name, **kwargs)
        self.destination_channel = destination_channel
        
    def build(self, input_shape):
        if self.destination_channel is None:
            self.destination_channel = input_shape[-1]
        # Halve input channels for compress_input conv
        self.compress_input = keras.layers.Conv2D(
            int(input_shape[-1] / 2), kernel_size=1, padding='same',
            activation=tf.nn.leaky_relu,
            kernel_initializer=tf.initializers.HeNormal(),
            bias_initializer=tf.initializers.HeUniform())
        
        self.conv = keras.layers.Conv2D(
            input_shape[-1], kernel_size=3, padding='same',
            activation=tf.nn.leaky_relu,
            kernel_initializer=tf.initializers.HeNormal(),
            bias_initializer=tf.initializers.HeUniform())
        
        self.pool = keras.layers.MaxPool2D(pool_size=3, strides=1, padding="same")
        
        self.compress_output = keras.layers.Conv2D(
            self.destination_channel, kernel_size=1, padding='same',
            activation=tf.nn.leaky_relu,
            kernel_initializer=tf.initializers.HeNormal(),
            bias_initializer=tf.initializers.HeUniform())
        
        super().build(input_shape)

    @tf.function
    def call(self, inputs, destination_size):
        # destination_size is expected to be a tensor of shape (2,) [height, width]
        compressed_input = self.compress_input(inputs)
        conv = self.conv(compressed_input)
        pool = self.pool(inputs)

        # tf.image.resize requires destination_size as int32 tensor or list with 2 elements
        # preserve_aspect_ratio=True sometimes causes issues in tf.function with tensor sizes
        # The original issue was the resize call with tensor size in tf.function
        # Here we keep preserve_aspect_ratio=True as in original code to stay faithful and test fix
        scaled_conv = tf.image.resize(conv, destination_size, preserve_aspect_ratio=True, antialias=True)
        scaled_pool = tf.image.resize(pool, destination_size, preserve_aspect_ratio=True, antialias=True)

        # Concatenate along channel axis
        concat = keras.layers.concatenate([scaled_pool, scaled_conv])
        compressed_output = self.compress_output(concat)
        return compressed_output

    def get_config(self):
        config = super().get_config()
        config.update({'destination_channel': self.destination_channel})
        return config


class Scaled1(keras.layers.Layer):
    def __init__(self, name="Scaled1", **kwargs):
        super().__init__(name=name, **kwargs)

    def build(self, input_shape):
        # input_shape expected: tuple of shapes: (res_shape, shc_shape)
        res_shape, shc_shape = input_shape
        self.scale_up = Scale(destination_channel=res_shape[-1])
        self.scale_down = Scale()
        super().build(input_shape)

    def call(self, inputs):
        inputs_res, inputs_shc = inputs
        
        # Use dynamic shape from tensor, as TensorShape may be incompatible inside tf.function
        shape1 = tf.shape(inputs_shc)[1:3]
        shape2 = tf.shape(inputs_shc)[1:3]

        scaled_res = self.scale_down(inputs_res, shape1)
        scaled_dense = self.scale_up(scaled_res, shape2)
        return scaled_dense      


class Scaled2(keras.layers.Layer):
    def __init__(self, name="Scaled2", **kwargs):
        super().__init__(name=name, **kwargs)

    def build(self, input_shape):
        res_shape, shc_shape = input_shape
        self.scale_up = Scale(destination_channel=res_shape[-1])
        self.scale_down = Scale()
        super().build(input_shape)

    def call(self, inputs):
        inputs_res, inputs_shc = inputs
        
        # Cast shapes explicitly to int32 for safety in tf.function
        shape1 = tf.cast(tf.shape(inputs_shc)[1:3], dtype=tf.int32)
        shape2 = tf.cast(tf.shape(inputs_shc)[1:3], dtype=tf.int32)

        scaled_res = self.scale_down(inputs_res, shape1)
        scaled_dense = self.scale_up(scaled_res, shape2)
        return scaled_dense


class Scaled3(keras.layers.Layer):
    def __init__(self, name="Scaled3", **kwargs):
        super().__init__(name=name, **kwargs)

    def build(self, input_shape):
        res_shape, shc_shape = input_shape
        self.scale_up = Scale(destination_channel=res_shape[-1])
        self.scale_down = Scale()
        super().build(input_shape)

    def call(self, inputs):
        inputs_res, inputs_shc = inputs
        
        # Access static shape - may cause issue inside tf.function (test case 3 in original)
        shape1 = inputs_shc.shape[1:3]
        shape2 = inputs_shc.shape[1:3]

        scaled_res = self.scale_down(inputs_res, shape1)
        scaled_dense = self.scale_up(scaled_res, shape2)
        return scaled_dense


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # For demonstration, we instantiate both Scale and use Scaled2 wrapper as example
        # The issue mostly concerns Scale usage, so we wrap Scaled2 inside MyModel
        self.scaled2 = Scaled2(name="Scaled2_wrapper")

    def call(self, inputs):
        # inputs expected as tuple of two tensors: (inputs_res, inputs_shc)
        return self.scaled2(inputs)


def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a tuple of two inputs matching inputs_res, inputs_shc
    # Following original example shapes in tests:
    # inputs_res: (1, 100, 100, 20)
    # inputs_shc: (1, 80, 80, 20)
    inputs_res = tf.random.uniform((1, 100, 100, 20), dtype=tf.float32)
    inputs_shc = tf.random.uniform((1, 80, 80, 20), dtype=tf.float32)
    return (inputs_res, inputs_shc)

