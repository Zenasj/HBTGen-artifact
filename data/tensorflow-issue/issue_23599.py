# tf.constant shape (4, 4), slicing [:, :2] -> shape (4, 2)

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Initialize a constant tensor of shape (4,4)
        # Same values as in the original example from the issue
        self.data = tf.constant(
            [[0, 1, 2, 3],
             [4, 5, 6, 7],
             [8, 9, 10, 11],
             [12, 13, 14, 15]],
            dtype=tf.float32,
            name='data')
    
    def call(self, inputs):
        # The original code sliced the constant: data_slice = data[:, :2]
        # Input is unused for this slicing operation but must be accepted to keep signature compatible.
        data_slice = self.data[:, :2]
        # Return sliced tensor
        return data_slice

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # The model’s forward method does not use input values but expects shape compatible with placeholder [4,2].
    # We'll provide random uniform tensor with shape (4, 2) and dtype float32 to match that.
    return tf.random.uniform((4, 2), dtype=tf.float32)

