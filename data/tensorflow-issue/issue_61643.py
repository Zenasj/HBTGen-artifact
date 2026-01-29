# tf.random.uniform((2, 2), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, n=2):
        super().__init__()
        # Since the user-tested code uses "RepeatVector" with a very large n,
        # but this causes a crash/error, we limit n to a small safe value by default.
        # The idea: repeat the input vectors n times along a new axis.
        self.n = n
        self.repeat_vector = tf.keras.layers.RepeatVector(self.n)
    
    def call(self, inputs):
        # inputs shape: (batch_size, features)
        # output shape: (batch_size, n, features)
        return self.repeat_vector(inputs)

def my_model_function():
    # Return an instance of MyModel.
    # Here we use a reasonable default n=2 (repeat 2 times).
    # The issue shows extremely large n causes failure/crash.
    return MyModel(n=2)

def GetInput():
    # Return a suitable input tensor for MyModel
    # Original repro uses shape [2, 2] with dtype float32.
    # This shape matches the expected input to RepeatVector.
    return tf.random.uniform(shape=(2, 2), dtype=tf.float32)

