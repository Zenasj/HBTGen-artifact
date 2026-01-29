# tf.random.normal([input_dim, output_size]), tf.zeros([output_size]) ← inferred input shapes from example

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, input_dim=4, output_size=3):
        super().__init__()
        # Variable weight matrix initialized randomly
        self.w = tf.Variable(
            tf.random.normal([input_dim, output_size]), name='w'
        )
        # Variable bias vector initialized to zeros
        self.b = tf.Variable(tf.zeros([output_size]), name='b')
        # Immediately override 'b' Variable with a constant tensor (non-trackable)
        # to demonstrate assigned non-trackable overwriting trackable attribute
        self.b = tf.constant(self.b.numpy())

    def call(self, x):
        # Simple linear layer: x * w + b
        # 'b' here is a constant tensor, not tracked as Variable
        return tf.matmul(x, self.w) + self.b

def my_model_function():
    # Create an instance of MyModel with default input_dim=4 and output_size=3
    model = MyModel()
    return model

def GetInput():
    # Return a random input tensor with shape matching input_dim of w (4)
    # Batch size (B) = 2 for demonstration, feature width (W) = 4
    # Using float32 as datatype consistent with tf.random.normal
    return tf.random.uniform((2, 4), dtype=tf.float32)

