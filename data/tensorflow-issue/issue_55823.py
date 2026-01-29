# tf.random.uniform((B, 3), dtype=tf.complex64) ← Based on example input shape (batch size arbitrary, width 3), complex64 dtype

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # This model replicates the BasicLayer from the issue:
        # a simple learned complex64 weight vector of shape (3,)
        self.w = self.add_weight(
            name='w',
            shape=(3,),
            initializer=tf.keras.initializers.Zeros(),
            dtype=tf.complex64,
            trainable=True,
        )

    def call(self, inputs):
        # Element-wise multiply inputs by the learned weight vector
        return inputs * self.w

def my_model_function():
    # Return an instance of MyModel with initialized weights
    return MyModel()

def GetInput():
    # Generate a random complex64 input tensor with shape (batch_size, 3)
    # Batch size is arbitrarily chosen here as 1, matching issue example
    real = tf.random.uniform((1, 3), minval=-1.0, maxval=1.0, dtype=tf.float32)
    imag = tf.random.uniform((1, 3), minval=-1.0, maxval=1.0, dtype=tf.float32)
    return tf.complex(real, imag)

