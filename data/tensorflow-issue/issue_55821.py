# tf.random.uniform((1, 512, 64, 7), dtype=tf.float32) ← Assumed batch size 1 for input shape (512,64,7)

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, UpSampling2D, MaxPool2D
from tensorflow.keras.activations import selu, sigmoid

# Custom layers and blocks rewritten to be compatible with TF 2.20 and for correct functional behavior


class up_conv(tf.keras.Model):
    """
    Up Convolution Block: Upsamples and applies Conv+BatchNorm+SELU activation.
    """

    def __init__(self, out_ch):
        super().__init__()
        self.up = tf.keras.Sequential([
            UpSampling2D(size=(2, 2)),
            Conv2D(out_ch, kernel_size=3, strides=1, padding='same', use_bias=True),
            BatchNormalization(momentum=0.97)
        ])

    def call(self, x):
        x = self.up(x)
        x = selu(x)
        return x


class Recurrent_block(tf.keras.Model):
    """
    Recurrent Block with iterations of Conv + BatchNorm + SELU.
    """

    def __init__(self, out_ch, t=2):
        super().__init__()
        self.t = t
        self.out_ch = out_ch
        self.conv = tf.keras.Sequential([
            Conv2D(out_ch, kernel_size=3, strides=1, padding='same', use_bias=True),
            BatchNormalization(momentum=0.97)
        ])

    def call(self, x):
        for i in range(self.t):
            if i == 0:
                x1 = self.conv(x)
            x1 = self.conv(x + x1)
            x1 = selu(x1)
        return x1


class Attention_block(tf.keras.Model):
    """
    Attention block that calculates attention map and scales input features.
    """

    def __init__(self, F_int):
        super().__init__()

        self.W_g = tf.keras.Sequential([
            Conv2D(F_int, kernel_size=1, strides=1, padding="same", use_bias=True),
            BatchNormalization(momentum=0.97)
        ])

        self.W_x = tf.keras.Sequential([
            Conv2D(F_int, kernel_size=1, strides=1, padding="same", use_bias=True),
            BatchNormalization(momentum=0.97)
        ])

        self.psi = tf.keras.Sequential([
            Conv2D(1, kernel_size=1, strides=1, padding="same", use_bias=True),
            BatchNormalization(momentum=0.97)
        ])

    def call(self, inputs):
        g, x = inputs
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = selu(g1 + x1)
        psi = self.psi(psi)
        psi = sigmoid(psi)
        out = x * psi
        return out


class RRCNN_block(tf.keras.Model):
    """
    Recurrent Residual CNN block consisting of a 1x1 Conv and two Recurrent_blocks,
    with residual connection.
    """

    def __init__(self, out_ch, t=2):
        super().__init__()
        self.Conv = Conv2D(out_ch, kernel_size=1, strides=1, padding="same")
        self.RCNN1 = Recurrent_block(out_ch, t=t)
        self.RCNN2 = Recurrent_block(out_ch, t=t)

    def call(self, x):
        x1 = self.Conv(x)
        x2 = self.RCNN1(x1)
        x2 = self.RCNN2(x2)
        out = x1 + x2
        return out


class MyModel(tf.keras.Model):
    """
    Residual Recurrent Convolutional Neural Network with Attention U-Net architecture.
    Adapted from R2AttU_Net, matching original input shape (512,64,7).
    """

    def __init__(self, out_ch=1, t=2):
        super().__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = MaxPool2D(pool_size=(2, 2), strides=2)
        self.Maxpool2 = MaxPool2D(pool_size=(2, 2), strides=2)
        self.Maxpool3 = MaxPool2D(pool_size=(2, 2), strides=2)
        self.Maxpool4 = MaxPool2D(pool_size=(2, 2), strides=2)

        self.RRCNN1 = RRCNN_block(filters[0], t=t)
        self.RRCNN2 = RRCNN_block(filters[1], t=t)
        self.RRCNN3 = RRCNN_block(filters[2], t=t)
        self.RRCNN4 = RRCNN_block(filters[3], t=t)
        self.RRCNN5 = RRCNN_block(filters[4], t=t)

        self.Up5 = up_conv(filters[3])
        self.Att5 = Attention_block(F_int=filters[2])
        self.Up_RRCNN5 = RRCNN_block(filters[3], t=t)

        self.Up4 = up_conv(filters[2])
        self.Att4 = Attention_block(F_int=filters[1])
        self.Up_RRCNN4 = RRCNN_block(filters[2], t=t)

        self.Up3 = up_conv(filters[1])
        self.Att3 = Attention_block(F_int=filters[0])
        self.Up_RRCNN3 = RRCNN_block(filters[1], t=t)

        self.Up2 = up_conv(filters[0])
        self.Att2 = Attention_block(F_int=32)  # As in original code
        self.Up_RRCNN2 = RRCNN_block(filters[0], t=t)

        self.Conv = Conv2D(out_ch, kernel_size=1, strides=1, padding="same")

    def call(self, x):
        # Encoder path
        e1 = self.RRCNN1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.RRCNN2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.RRCNN3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.RRCNN4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.RRCNN5(e5)

        # Decoder path with attention and upsampling
        d5 = self.Up5(e5)
        e4_att = self.Att5([d5, e4])
        d5 = tf.concat([e4_att, d5], axis=-1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        e3_att = self.Att4([d4, e3])
        d4 = tf.concat([e3_att, d4], axis=-1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        e2_att = self.Att3([d3, e2])
        d3 = tf.concat([e2_att, d3], axis=-1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        e1_att = self.Att2([d2, e1])
        d2 = tf.concat([e1_att, d2], axis=-1)
        d2 = self.Up_RRCNN2(d2)

        out = self.Conv(d2)
        return out


def my_model_function():
    # Instantiate MyModel with default out channels=1, recurrence depth=2
    return MyModel()


def GetInput():
    # Create a random input tensor matching input shape used in original model:
    # [batch_size, height=512, width=64, channels=7]
    # Using batch size = 1 for example.
    return tf.random.uniform((1, 512, 64, 7), dtype=tf.float32)

