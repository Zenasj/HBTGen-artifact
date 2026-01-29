# tf.random.uniform((1, 5), dtype=tf.float32) ← Input shape is (batch_size=1, 5) as per the original keras.Input

import tensorflow as tf


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # The model mimics the reported model:
        # Input shape (1,5)
        # Two Dense layers of 5 units each (no activation specified, linear)
        # Then a floor modulus by 3 operation as a Lambda layer
        
        # Note: The original Sequential model had this:
        # tf.keras.Input(shape=(5), batch_size=1), which means input shape (5,)
        # but the first Dense layer uses input_shape=(1,5) - which is inconsistent.
        # To reconcile, accept input shape (1,5), i.e. batch=1 and input shape (1,5).
        # But the Keras Input used shape=(5), batch_size=1,
        # meaning input tensor shape = (1, 5).
        # The model is sequential, so input shape to first Dense is (5,)
        # Let's assume input shape (1,5) for batch of size 1 as 2D tensor (1,5).
        #
        # The first Dense layer expects input_shape (1,5) but Dense by default expects 2D input.
        # We'll maintain input shape (1,5) for batch size 1, consistent with input.
        
        self.dense1 = tf.keras.layers.Dense(5)
        self.dense2 = tf.keras.layers.Dense(5)
        # Since floor_mod is not a native keras layer, use Lambda to mimic the floormod by 3
        self.floormod = tf.keras.layers.Lambda(lambda x: tf.math.floormod(x, 3))

    @tf.function(jit_compile=True)
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.floormod(x)
        return x


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Create a random tensor compatible with the input shape expected by MyModel:
    # The model input shape is (1,5) as per tf.keras.Input(shape=(5), batch_size=1)
    # We'll generate a tensor of shape (1,5) with dtype float32 (for Dense layers)
    return tf.random.uniform((1, 5), dtype=tf.float32)

