# tf.constant with shape (3,) as input dictionary values for features 'first_feature' and 'second_feature'
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # This model does not really depend on inputs - it returns a random tensor of shape (3, 1)
        # This mimics the example from the issue where output is tf.random.uniform([3, 1])
    
    def call(self, inputs, training=None):
        # inputs is expected to be a dictionary with keys 'first_feature' and 'second_feature',
        # each a tensor of shape (3,) as per the example
        return tf.random.uniform([3, 1], dtype=tf.float32)

def my_model_function():
    # Instantiate MyModel - no weights or special initialization needed
    return MyModel()

def GetInput():
    # Provide a dictionary input with keys 'first_feature' and 'second_feature'
    # Each is a tensor of shape (3,), matching the example in the GitHub issue
    # Values are int32 as in the example [1,2,3] and [4,5,6]
    return {
        "first_feature": tf.constant([1, 2, 3], dtype=tf.int32),
        "second_feature": tf.constant([4, 5, 6], dtype=tf.int32),
    }

