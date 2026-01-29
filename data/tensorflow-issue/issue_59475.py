# tf.random.uniform((1875,), dtype=tf.int32) ← input shape inferred from np.array of length 1875 used in the loop

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # No trainable parameters or layers needed for this simple condition check
    
    def sum_func(self, counter):
        # sum_func equivalent implemented as pure tf operation
        return tf.equal(tf.reduce_sum(counter), 1875)
    
    @tf.function
    def call(self, counter):
        # Equivalent of tf_func in user code
        return self.sum_func(counter)

def my_model_function():
    # Return an instance of MyModel; no special initialization required
    return MyModel()

def GetInput():
    # Return a 1-D int32 tensor of length 1875 filled with ones to represent the input array "counter"
    # as built in the original code by appending 1 each step up to 1875 elements.
    return tf.ones(shape=(1875,), dtype=tf.int32)

