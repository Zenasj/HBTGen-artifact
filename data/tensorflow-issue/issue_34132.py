# tf.random.uniform((1, 128, 128, 3), dtype=tf.float32)
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU

def get_norm(norm_type):
    if norm_type == "batch":
        return BatchNormalization()
    else:
        raise ValueError(f"Unrecognized norm_type {norm_type}")

class MyModel(tf.keras.Model):
    def __init__(self,
                 base_filters=32,
                 lrelu_alpha=0.2,
                 pad_type="same",
                 norm_type="batch"):
        super(MyModel, self).__init__(name="Discriminator")
        # 1------------------------------------------
        self.conv1 = Conv2D(
                filters=base_filters, # 32
                kernel_size=3,
                padding=pad_type
                )
        self.relu1 = LeakyReLU(alpha=lrelu_alpha)

        # 2-------------------------------------------
        self.conv2a = Conv2D(
                filters=base_filters*2, # 64
                strides=2,
                kernel_size=3,
                padding=pad_type
                )
        self.relu2a = LeakyReLU(alpha=lrelu_alpha)
        self.conv2b = Conv2D(
                filters=base_filters*4, # 128
                kernel_size=3,
                padding=pad_type
                )
        self.norm2 = get_norm(norm_type)
        self.relu2b = LeakyReLU(alpha=lrelu_alpha)

        # 3-----------------------------------------------
        self.conv3a = Conv2D(
                filters=base_filters*4, # 128
                strides=2,
                kernel_size=3,
                padding=pad_type
                )
        self.relu3a = LeakyReLU(alpha=lrelu_alpha)
        self.conv3b = Conv2D(
                filters=base_filters*8, # 256
                kernel_size=3,
                padding=pad_type
                )
        self.norm3 = get_norm(norm_type)
        self.relu3b = LeakyReLU(alpha=lrelu_alpha)

        # 4----------------------------------------------
        self.conv4 = Conv2D(
                filters=base_filters*8, # 256
                kernel_size=3,
                padding=pad_type
                )
        self.norm4 = get_norm(norm_type)
        self.relu4 = LeakyReLU(alpha=lrelu_alpha)

        # final--------------------------------------------
        self.conv_final = Conv2D(
                filters=1,
                kernel_size=3,
                padding=pad_type
                )

    def build(self, input_shape):
        super(MyModel, self).build(input_shape)

    def call(self, input_tensor, training=False):
        # 1------------------------------------------
        x = self.conv1(input_tensor, training=training)
        x = self.relu1(x)

        # 2------------------------------------------
        x = self.conv2a(x, training=training)
        x = self.relu2a(x)
        x = self.conv2b(x, training=training)
        x = self.norm2(x, training=training)
        x = self.relu2b(x)

        # 3------------------------------------------
        x = self.conv3a(x, training=training)
        x = self.relu3a(x)
        x = self.conv3b(x, training=training)
        x = self.norm3(x, training=training)
        x = self.relu3b(x)

        # 4------------------------------------------
        x = self.conv4(x, training=training)
        x = self.norm4(x, training=training)
        x = self.relu4(x)

        # final----------------------------------------
        x = self.conv_final(x, training=training)

        return x

def my_model_function():
    # Instantiate and return the model
    return MyModel()

def GetInput():
    # Return a random tensor input matching input expected by MyModel
    # Based on example: batch=1, height=128, width=128, channels=3, dtype float32
    return tf.random.uniform((1, 128, 128, 3), dtype=tf.float32)

