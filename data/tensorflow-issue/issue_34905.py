# tf.random.uniform((B,), dtype=...) <- Input is a 1D tensor of varying length, as in np.ones(i) calls with varying i

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # No additional layers needed for a simple reduce_sum operation
        
    @tf.function(experimental_relax_shapes=True)
    def call(self, data):
        # This replicates the behavior of the example's instance method wrapped with experimental_relax_shapes
        return tf.reduce_sum(data)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate a random 1D tensor of variable length to simulate input like np.ones(i)
    # We pick a random length between 1 and 10 for demonstration
    length = tf.random.uniform(shape=(), minval=1, maxval=11, dtype=tf.int32)
    return tf.ones(shape=(length,), dtype=tf.float32)

