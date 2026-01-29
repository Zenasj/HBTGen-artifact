# tf.random.uniform((1, 64, 64, 8), dtype=tf.float32) ← input tensor shape and dtype used in Infer.infer_ input_signature

import tensorflow as tf

class Inner(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            8,
            (3, 3),
            kernel_initializer=tf.keras.initializers.he_normal(),
            padding='same',
            name='conv1'
        )
    def call(self, x, dummy=False):
        # The dummy argument is unused but retained for compatibility
        x = self.conv1(x)
        return x

class Outer(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.down = tf.keras.layers.Conv2D(
            8,
            (3, 3),
            strides=(2, 2),
            kernel_initializer=tf.keras.initializers.he_normal(),
            padding='same',
            name='down'
        )
        self.inner = Inner()

    def call(self, x, dummy=False):
        x_small = self.down(x)
        # Call inner model on original input and downsampled input
        out1 = self.inner(x, dummy=dummy)
        out2 = self.inner(x_small, dummy=dummy)
        # Return both outputs as a tuple
        return out1, out2

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.outer = Outer()

    def call(self, x):
        return self.outer(x)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input matching expected shape (batch=1, 64x64 spatial, 8 channels)
    # dtype float32 matches the input_signature in the original code
    return tf.random.uniform((1, 64, 64, 8), dtype=tf.float32)

