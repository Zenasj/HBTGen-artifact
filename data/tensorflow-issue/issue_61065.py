# tf.constant([1.], shape=[1,1]) ← Input shape is [1,1], single float value tensor
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # No trainable parameters or layers needed for this simple model

    def call(self, x1):
        # As per the original code, return three identity matrices of different sizes
        x2 = tf.eye(1, dtype=tf.float32)  # Shape (1,1)
        x3 = tf.eye(2, dtype=tf.float32)  # Shape (2,2)
        x4 = tf.eye(1, dtype=tf.float32)  # Shape (1,1)
        return [x2, x3, x4]

def my_model_function():
    # Returns an instance of MyModel
    return MyModel()

def GetInput():
    # Returns a tensor matching input shape [1,1]
    # Use dtype=tf.float32 to be consistent with model inputs
    return tf.constant([1.], shape=[1,1], dtype=tf.float32)

