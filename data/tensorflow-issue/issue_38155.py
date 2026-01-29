# tf.random.uniform((B, H, W, C), dtype=tf.float32) where B=500, H=32, W=32, C=3

import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()

        # Convolution layer 1:
        self.conv1 = layers.Conv2D(
            filters=32,
            kernel_size=[5, 5],
            strides=(1, 1),
            padding='same',
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
            bias_initializer=tf.zeros_initializer(),
            data_format='channels_last'
        )
        # Max pooling layer 1:
        self.pool1 = layers.MaxPool2D(
            pool_size=[3, 3], strides=2,
            data_format='channels_last',
            padding='VALID'
        )

        # Convolution layer 2:
        self.conv2 = layers.Conv2D(
            filters=64,
            kernel_size=[5, 5],
            strides=(1,1),
            kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
            bias_initializer=tf.zeros_initializer(),
            padding='same',
            activation=tf.nn.relu,
            data_format='channels_last'
        )
        # Max pooling layer 2:
        self.pool2 = layers.MaxPool2D(
            pool_size=[3, 3], strides=2,
            data_format='channels_last',
            padding="VALID"
        )

        # Flatten layer to reshape pooled output before dense layers:
        self.flatten = layers.Flatten()

        # Dense layer 1:
        self.dense1 = layers.Dense(
            units=784,
            activation=tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.05),
            bias_initializer=tf.zeros_initializer()
        )
        # Batch normalization:
        self.batchNorm = layers.BatchNormalization()

        # Dense layer 2 (output layer - 10 classes):
        self.dense2 = layers.Dense(
            units=10,
            activation=tf.nn.relu,  # Note: In practice, softmax preferred for classification output
            kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.05),
            bias_initializer=tf.zeros_initializer()
        )

    def call(self, inputs):
        # Print statements demonstrate shape inside the call method.
        # inputs.shape[0] will be None when tracing or in eager mode - 
        # but will show batch size dynamically at runtime.
        print("Inside MyModel.call()")
        print(f"inputs.shape (static): {inputs.shape}")  # e.g. (None, 32, 32, 3)
        print(f"tf.shape(inputs) (dynamic): {tf.shape(inputs)}")  # e.g. [batch,32,32,3]

        x = self.conv1(inputs)   # (batch, 32, 32, 32)
        x = self.pool1(x)        # (batch, 15, 15, 32)
        x = self.conv2(x)        # (batch, 15, 15, 64)
        x = self.pool2(x)        # (batch, 7, 7, 64)
        x = self.flatten(x)      # (batch, 7*7*64)
        x = self.dense1(x)       # (batch, 784)
        x = self.batchNorm(x)    # (batch, 784)
        x = self.dense2(x)       # (batch, 10)

        # Note: the original example used relu activation on final layer,
        # softmax activation would be typical for classification outputs,
        # but we keep relu to stay faithful to original code.
        return x

def my_model_function():
    """
    Returns an instance of MyModel.
    No weights initialization from checkpoints done here,
    only default initializers as per layers.
    """
    return MyModel()

def GetInput():
    """
    Returns a tensor shaped (500, 32, 32, 3) of random floats,
    to simulate a batch of 500 CIFAR-10 images.
    dtype: tf.float32 to match typical image inputs.
    """
    batch_size = 500
    height = 32
    width = 32
    channels = 3
    # Random data simulating a batch of images
    return tf.random.uniform((batch_size, height, width, channels), dtype=tf.float32)

