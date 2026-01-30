from tensorflow import keras
from tensorflow.keras import layers

# from tensorflow import math
import tensorflow as tf
import keras
# from tensorflow.keras import Model
from keras import Model, Input
from keras.layers import (Concatenate, Lambda, UpSampling2D, MaxPool2D,
                          ZeroPadding2D, Conv2D, BatchNormalization)
from keras.activations import selu as SeLU
from keras.activations import sigmoid as Sigmoid


# from utils.utils import compose


# def gelu_(X):
#     return 0.5 * X * (1.0 + math.tanh(0.7978845608028654 * (X + 0.044715 * math.pow(X, 3))))
#
#
# def snake_(X, beta):
#     return X + (1 / beta) * math.square(math.sin(beta * X))
#
#
# class GELU(Model):
#     '''
#     Gaussian Error Linear Unit (GELU), an alternative of ReLU
#
#     Y = GELU()(X)
#
#     ----------
#     Hendrycks, D. and Gimpel, K., 2016. Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415.
#
#     Usage: use it as a tf.keras.Model
#
#
#     '''
#
#     def __init__(self, trainable=False, **kwargs):
#         super(GELU, self).__init__(**kwargs)
#         self.supports_masking = True
#         self.trainable = trainable
#
#     def build(self, input_shape):
#         super(GELU, self).build(input_shape)
#
#     def call(self, inputs, mask=None):
#         return gelu_(inputs)
#
#     def get_config(self):
#         config = {'trainable': self.trainable}
#         base_config = super(GELU, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))
#
#     def compute_output_shape(self, input_shape):
#         return input_shape


class up_conv(Model):
    """
    Up Convolution Block
    """

    def __init__(self, out_ch):
        super(up_conv, self).__init__()
        self.up = keras.Sequential()
        self.up.add(UpSampling2D(size=(2, 2)))
        self.up.add(Conv2D(out_ch, kernel_size=3, strides=1, padding='same', use_bias=True))
        self.up.add(BatchNormalization(momentum=0.97))

    def call(self, x, **kwargs):
        x = self.up(x)
        x = SeLU(x)
        return x


class Recurrent_block(Model):
    """
    Recurrent Block for R2Unet_CNN
    """

    def __init__(self, out_ch, t=2):
        super(Recurrent_block, self).__init__()

        self.t = t
        self.out_ch = out_ch
        self.conv = keras.Sequential()
        self.conv.add(Conv2D(out_ch, kernel_size=3, strides=1, padding='same', use_bias=True))
        self.conv.add(BatchNormalization(momentum=0.97))

    def call(self, x, **kwargs):
        for i in range(self.t):
            if i == 0:
                x = self.conv(x)
            out = self.conv(x + x)
            out = SeLU(out)
        return out


class Attention_block(Model):
    """
    Attention Block
    """

    def __init__(self, F_int):
        super(Attention_block, self).__init__()

        self.W_g = keras.Sequential()
        self.W_g.add(Conv2D(F_int, kernel_size=1, strides=1, padding="same", use_bias=True)),
        self.W_g.add(BatchNormalization(momentum=0.97))

        self.W_x = keras.Sequential()
        self.W_x.add(Conv2D(F_int, kernel_size=1, strides=1, padding="same", use_bias=True))
        self.W_x.add(BatchNormalization(momentum=0.97))

        self.psi = keras.Sequential()
        self.psi.add(Conv2D(1, kernel_size=1, strides=1, padding="same", use_bias=True)),
        self.psi.add(BatchNormalization(momentum=0.97))

    def call(self, inputs, **kwargs):
        g, x = inputs
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = SeLU(g1 + x1)
        psi = self.psi(psi)
        psi = Sigmoid(psi)
        out = x * psi
        return out


class RRCNN_block(Model):
    """
    Recurrent Residual Convolutional Neural Network Block
    """

    def __init__(self, out_ch, t=2):
        super(RRCNN_block, self).__init__()

        self.Conv = Conv2D(out_ch, kernel_size=1, strides=1, padding="same")
        # self.RCNN = compose(
        #     Recurrent_block(out_ch, t=t),
        #     Recurrent_block(out_ch, t=t)
        # )
        self.RCNN1 = Recurrent_block(out_ch, t=t)
        self.RCNN2 = Recurrent_block(out_ch, t=t)

    def call(self, x, **kwargs):
        x1 = self.Conv(x)
        # x2 = self.RCNN(x1)
        x2 = self.RCNN1(x1)
        x2 = self.RCNN2(x2)
        out = x1 + x2
        return out


class R2AttU_Net(Model):
    """
    Residual Recuurent Block with attention Unet
    Implementation : https://github.com/LeeJunHyun/Image_Segmentation
    """

    def __init__(self, out_ch=1, t=2):
        super(R2AttU_Net, self).__init__()

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
        self.Att2 = Attention_block(F_int=32)
        self.Up_RRCNN2 = RRCNN_block(filters[0], t=t)

        self.Conv = Conv2D(out_ch, kernel_size=1, strides=1, padding="same")

    def call(self, x, **kwargs):
        e1 = self.RRCNN1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.RRCNN2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.RRCNN3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.RRCNN4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.RRCNN5(e5)

        d5 = self.Up5(e5)
        # print("d5.shape:{}, e4.shape{}".format(d5.shape, e4.shape))
        e4 = self.Att5([d5, e4])
        d5 = tf.concat((e4, d5), axis=-1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        # print("d4.shape:{}, e3.shape{}".format(d4.shape, e3.shape))
        e3 = self.Att4([d4, e3])
        d4 = tf.concat((e3, d4), axis=-1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        e2 = self.Att3([d3, e2])
        d3 = tf.concat((e2, d3), axis=-1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        e1 = self.Att2([d2, e1])
        d2 = tf.concat((e1, d2), axis=-1)
        d2 = self.Up_RRCNN2(d2)

        out = self.Conv(d2)

        return out


if __name__ == '__main__':
    inputs1 = Input([512, 64, 7])
    inputs2 = Input([None, None, 1])
    # x1 = up_conv(1, 3)(inputs)
    # print(x1)
    # x2 = Attention_block(3, )(inputs1, inputs2)
    # print(x2)
    # x3 = RRCNN_block(3)(inputs1)
    # print(x3)
    deep_t = 2
    x3 = R2AttU_Net(1, deep_t)(inputs1)
    print(x3)