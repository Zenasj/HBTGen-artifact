# tf.random.uniform((B, 64, 64, 3), dtype=tf.float32) ← Input shape inferred from SRGAN generator input

import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        Gen_init = tf.keras.initializers.HeNormal()
        axis = -1
        shared_axes = [1, 2]

        # Pre-residual block conv + PReLU
        self.conv1 = layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1),
                                   padding='same', kernel_initializer=Gen_init)
        self.prelu1 = layers.PReLU(alpha_initializer='zeros',
                                  alpha_regularizer=None,
                                  alpha_constraint=None,
                                  shared_axes=shared_axes)
        # Residual block layers
        self.res_conv1 = layers.Conv2D(64, kernel_size=(3,3), strides=(1,1),
                                      padding='same', use_bias=False,
                                      kernel_initializer=Gen_init, activation=None)
        self.res_bn1 = layers.BatchNormalization(axis=axis)
        self.res_prelu = layers.PReLU(alpha_initializer='zeros',
                                     alpha_regularizer=None,
                                     alpha_constraint=None,
                                     shared_axes=shared_axes)
        self.res_conv2 = layers.Conv2D(64, kernel_size=(3,3), strides=(1,1),
                                      padding='same', use_bias=False,
                                      kernel_initializer=Gen_init, activation=None)
        self.res_bn2 = layers.BatchNormalization(axis=axis)

        # Post residual block conv + bn
        self.post_res_conv = layers.Conv2D(64, kernel_size=(3,3), strides=(1,1),
                                           padding='same', use_bias=False,
                                           kernel_initializer=Gen_init, activation=None)
        self.post_res_bn = layers.BatchNormalization(axis=axis)

        # Upsampling blocks
        self.up1 = layers.UpSampling2D(size=(2,2))
        self.up_conv1 = layers.Conv2D(256, kernel_size=(3,3), strides=(1,1),
                                      padding='same', use_bias=False,
                                      kernel_initializer=Gen_init, activation=None)
        self.up_prelu1 = layers.PReLU(alpha_initializer='zeros',
                                     alpha_regularizer=None,
                                     alpha_constraint=None,
                                     shared_axes=shared_axes)

        self.up2 = layers.UpSampling2D(size=(2,2))
        self.up_conv2 = layers.Conv2D(256, kernel_size=(3,3), strides=(1,1),
                                      padding='same', use_bias=False,
                                      kernel_initializer=Gen_init, activation=None)
        self.up_prelu2 = layers.PReLU(alpha_initializer='zeros',
                                     alpha_regularizer=None,
                                     alpha_constraint=None,
                                     shared_axes=shared_axes)

        # Final conv layer with tanh activation (output range [-1,1])
        self.final_conv = layers.Conv2D(3, kernel_size=(9,9), strides=(1,1),
                                        padding='same', use_bias=False,
                                        kernel_initializer=Gen_init,
                                        activation='tanh')

        # Add layers shortcut for residual connection
        self.add = layers.Add()

    def call(self, inputs, training=False):
        # x: Input tensor of shape (B, 64, 64, 3)
        x = self.conv1(inputs)
        x_input_res_block = self.prelu1(x)

        # Residual block
        res_x = self.res_conv1(x_input_res_block)
        res_x = self.res_bn1(res_x, training=training)
        res_x = self.res_prelu(res_x)
        res_x = self.res_conv2(res_x)
        res_x = self.res_bn2(res_x, training=training)

        # Add skip connection inside residual block
        res_x = self.add([res_x, x_input_res_block])

        # Post residual block conv + bn
        x = self.post_res_conv(res_x)
        x = self.post_res_bn(x, training=training)

        # Skip connection with x_input_res_block
        x = self.add([x, x_input_res_block])

        # Two upsampling blocks (scale x4)
        x = self.up1(x)
        x = self.up_conv1(x)
        x = self.up_prelu1(x)

        x = self.up2(x)
        x = self.up_conv2(x)
        x = self.up_prelu2(x)

        # Final conv with tanh activation
        output = self.final_conv(x)

        return output


def my_model_function():
    # Return an instance of MyModel
    return MyModel()


def GetInput():
    # Return a random input tensor with batch size 1 and shape (64,64,3), dtype float32.
    # This matches model input shape from the example.
    return tf.random.uniform((1, 64, 64, 3), dtype=tf.float32)

