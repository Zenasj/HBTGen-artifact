# tf.random.uniform((10,), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # The original code was TF1 style graph session code with placeholder input shape [10].
        # For TF2, we just accept input tensors of shape [10].
        # The model replicates: b = a + 1, c = b * 2.
    
    @tf.function(jit_compile=True)
    def call(self, inputs):
        # inputs: a tensor with shape (10,)
        b = inputs + 1
        c = b * 2
        # To keep behavior aligned to original code, return c as final output
        return c

def my_model_function():
    # Return an instance of MyModel; no custom initialization required
    return MyModel()

def GetInput():
    # Generate a random tensor with shape (10,) and dtype float32, matching 'a'
    return tf.random.uniform((10,), dtype=tf.float32)

