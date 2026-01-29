# tf.random.uniform((1, 1, 32, 25), dtype=tf.float32)  # Inferred input shape based on the PyTorch example input (N, C, H, W) for the analogous TF model

import tensorflow as tf

class PadLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(PadLayer, self).__init__()
        # Padding corresponds to PyTorch's (0,0,1,1) on last two dims.
        # TensorFlow padding format: [[batch], [height], [width], [channels]]
        # Here, input shape is [N, H, W, C], padding applied on height and width dims.
        # From PyTorch code, padding=(0,0,1,1) means pad bottom +1 and right +1 with value -1.
        self.paddings = tf.constant([[0, 0], [0, 1], [0, 1], [0, 0]])
        self.constant_values = -1.0

    def call(self, inputs):
        return tf.pad(inputs, self.paddings, mode='CONSTANT', constant_values=self.constant_values)

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Mimicking PyTorch Conv2d(in_channels=1, out_channels=16, kernel_size=(1,7), padding='same'):
        # TensorFlow Conv2D expects input shape NHWC and kernel shape [kernel_height, kernel_width, in_channels, out_channels]
        self.conv = tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=(1, 7),
            padding='same',
            data_format='channels_last',
            use_bias=True,
            activation=None
        )
        self.pad = PadLayer()
        # MaxPool2d(kernel_size=(1,3), stride=(1,2), padding=0) in PyTorch
        # TensorFlow MaxPool2D requires pool_size, strides as tuples along H,W dims
        self.mpool = tf.keras.layers.MaxPooling2D(pool_size=(1, 3), strides=(1, 2), padding='valid')

    def call(self, inputs):
        # inputs shape: (batch, height, width, channels) e.g. (1, 32, 25, 1)
        x = self.conv(inputs)
        x = self.pad(x)
        x = self.mpool(x)
        return x

def my_model_function():
    # Return an instance of MyModel; weights randomly initialized as no pretrained weights given.
    return MyModel()

def GetInput():
    # Return a random tensor input matching the expected input: (batch, height, width, channels) = (1, 32, 25, 1)
    return tf.random.uniform(shape=(1, 32, 25, 1), dtype=tf.float32)

