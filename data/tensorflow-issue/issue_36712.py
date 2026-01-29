# tf.random.uniform((B, H, W, C), dtype=tf.float32) ← Input shape inferred from the example: (None, 32, 32, 3)

import tensorflow as tf

class OpOrSkip(tf.keras.layers.Layer):
    def __init__(self, op):
        super().__init__()
        self.op = op
        
    def call(self, x):
        # Randomly decide to apply op or skip (return input)
        rnd = tf.random.uniform(())
        # Because tf.cond or python if-else inside tf.function can be tricky,
        # using tf.cond to ensure graph compatibility
        return tf.cond(rnd < 0.5,
                       lambda: self.op(x),
                       lambda: x)

def skip_conv(s):
    x = tf.keras.layers.Conv2D(3, 3, padding='same')(s)
    # Residual skip connection
    return x + s 

def func_as_model(func, shape):
    inp = tf.keras.Input(shape)
    out = func(inp)
    return tf.keras.Model(inputs=inp, outputs=out)

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Build the skip connection model
        self.skip_conv_model = func_as_model(skip_conv, [32, 32, 3])
        # Wrap the skip_conv_model inside OpOrSkip, which randomly skips the residual block
        self.op_or_skip = OpOrSkip(self.skip_conv_model)

    def call(self, inputs):
        # Forward pass through OpOrSkip layer using skip_conv_model internally
        return self.op_or_skip(inputs)

def my_model_function():
    # Create and return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor of shape [batch_size, 32, 32, 3]
    # Assuming batch size 4 for demonstration, dtype float32
    return tf.random.uniform((4, 32, 32, 3), dtype=tf.float32)

