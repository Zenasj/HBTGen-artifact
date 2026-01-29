# tf.random.uniform((batch_size, num_classes, 1), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # This model demonstrates a max_pool1d operation on input shape (N, W, C)
        # where N=batch_size, W=num_classes, C=1 in this example.

    def call(self, x):
        # x is expected to have shape (batch_size, width, channels)
        # Use max_pool1d with ksize=[width, 1, 1] and strides=[1, 1, 1] as per example.
        width = tf.shape(x)[1]

        # Use max_pool1d with data_format NWC (batch, width, channels)
        # ksize and strides are [width,1,1] and [1,1,1] respectively, 
        # so the max_pool will pool across the entire width dimension.
        y_max = tf.nn.max_pool1d(input=x,
                                 ksize=[width, 1, 1],
                                 strides=[1, 1, 1],
                                 padding='VALID',
                                 data_format='NWC')
        # y_max shape is (batch_size, 1, channels), reshape to (batch_size,)
        y_max = tf.reshape(y_max, [tf.shape(x)[0]])
        return y_max

def my_model_function():
    # Return an instance of MyModel without pretrained weights or special init.
    return MyModel()

def GetInput():
    # Based on the example in the issue:
    # Input shape (batch_size=2, width=num_classes=10, channels=1)
    batch_size = 2
    num_classes = 10
    channels = 1
    # Generate random floats for input tensor
    x = tf.random.uniform(shape=(batch_size, num_classes, channels), dtype=tf.float32)
    return x

