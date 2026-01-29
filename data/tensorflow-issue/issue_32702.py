# tf.random.uniform((B, 1), dtype=tf.float32) ← Based on input shape (1,) and example input np.ones((1,1), dtype=np.float32)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Model has two outputs:
        # out0: cast input to int32
        # out1: cast input to float64
        self.cast_int32 = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.int32))
        self.cast_float64 = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float64))

    def call(self, inputs):
        out0 = self.cast_int32(inputs)
        out1 = self.cast_float64(inputs)
        return [out0, out1]

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random input tensor matching the expected input shape and dtype
    # Input shape: (batch_size=1, 1), dtype float32 (as per example)
    return tf.random.uniform((1, 1), dtype=tf.float32)

