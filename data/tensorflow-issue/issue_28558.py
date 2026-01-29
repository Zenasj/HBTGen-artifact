# tf.random.normal((B, 100)) ← Input shape inferred from example noise input to make_generator_model()

import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self, name=None):
        super().__init__(name=name)
        # This is a reimplementation of the make_generator_model from the issue:
        # a generator model with Dense + BatchNorm + LeakyReLU + Conv2DTranspose layers.
        # Input shape is (batch_size, 100)
        #
        # To avoid name uniquification inconsistencies discussed in the issue,
        # we explicitly name each layer here with fixed names so that repeated instantiations
        # generate consistent variable names.
        #
        # This aligns with the recommendation from the issue comments to pass explicit `name` arguments.
        self.dense = layers.Dense(7*7*256, use_bias=False, input_shape=(100,), name='dense')
        self.batchnorm1 = layers.BatchNormalization(name='batch_normalization')
        self.leakyrelu1 = layers.LeakyReLU(name='leaky_relu_1')

        self.reshape = layers.Reshape((7, 7, 256), name='reshape')

        self.conv2dtranspose1 = layers.Conv2DTranspose(128, (5,5), strides=(1,1), padding='same', use_bias=False,
                                                       name='conv2d_transpose_1')
        self.batchnorm2 = layers.BatchNormalization(name='batch_normalization_1')
        self.leakyrelu2 = layers.LeakyReLU(name='leaky_relu_2')

        self.conv2dtranspose2 = layers.Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', use_bias=False,
                                                       name='conv2d_transpose_2')
        self.batchnorm3 = layers.BatchNormalization(name='batch_normalization_2')
        self.leakyrelu3 = layers.LeakyReLU(name='leaky_relu_3')

        self.conv2dtranspose3 = layers.Conv2DTranspose(1, (5,5), strides=(2,2), padding='same', use_bias=False,
                                                       name='conv2d_transpose_3', activation='tanh')

    def call(self, inputs, training=False):
        x = self.dense(inputs)
        x = self.batchnorm1(x, training=training)
        x = self.leakyrelu1(x)
        x = self.reshape(x)
        x = self.conv2dtranspose1(x)
        x = self.batchnorm2(x, training=training)
        x = self.leakyrelu2(x)
        x = self.conv2dtranspose2(x)
        x = self.batchnorm3(x, training=training)
        x = self.leakyrelu3(x)
        x = self.conv2dtranspose3(x)
        return x


def my_model_function():
    # Return an instance of MyModel with fixed layer names to avoid inconsistent variable naming.
    return MyModel()


def GetInput():
    # Return a random tensor input that matches the expected input of MyModel.
    # Batch size = 1, input shape = (100,)
    return tf.random.normal((1, 100))

