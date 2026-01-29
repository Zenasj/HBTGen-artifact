# tf.random.uniform((1, 2), dtype=tf.string) ← Input shape is (batch=1, length=2) with dtype string

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # The core issue is that tf.gather in TFLite does not support inputs with more than 1 dimension.
        # We reshape the input tensor from shape (1, 2) to (2,) before applying gather.
        # This matches the workaround described in the issue conversation.
        self.gather_indices = tf.constant([0])

    def call(self, inputs):
        # inputs shape: (batch=1, 2)
        # reshape to 1D [2] to satisfy TFLite gather requirements.
        reshaped = tf.reshape(inputs, shape=[2])
        gathered = tf.gather(reshaped, self.gather_indices)
        return gathered


def my_model_function():
    # Create and return an instance of MyModel
    return MyModel()


def GetInput():
    # Return a sample input tensor compatible with the model:
    # shape (1, 2)  - batch of one element with two strings
    sample_strings = tf.constant([[b'hello', b'world']], dtype=tf.string)
    return sample_strings

