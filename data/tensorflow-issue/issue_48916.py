# tf.random.uniform((B, 512, 512, 1), dtype=tf.float32) ← assumed input shape from network definition (512x512 grayscale image)

import tensorflow as tf
from tensorflow.keras.layers import SeparableConv2D, BatchNormalization, Activation, ReLU, MaxPool2D, PReLU

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # Parameters inferred from the description
        self.nrSH_in = 27  # number of input spherical harmonics coeff.
        self.baseFilter = 16
        self.nrSH_out = 9  # output SH
        self.ncPre = self.baseFilter

        self.ncHG3 = self.baseFilter
        self.ncHG2 = self.baseFilter * 2
        self.ncHG1 = self.baseFilter * 4
        self.ncHG0 = self.baseFilter * 8 + self.nrSH_in  # Bottleneck channels = 8*16+27 = 155
        
        # Initial layers
        self.pre_conv = SeparableConv2D(self.ncPre, kernel_size=5, strides=1, padding="same", name="pre_conv")
        self.pre_bn = BatchNormalization(name="pre_bn")
        self.relu_1 = ReLU()

        # HG3 block
        self.HG3_BB_Upper_conv1 = SeparableConv2D(16, kernel_size=3, strides=1, padding="same", use_bias=False)
        self.HG3_BN1 = BatchNormalization()

        self.HG3_BB_Upper_conv2 = SeparableConv2D(16, 3, 1, padding="same", use_bias=False)
        self.HG3_BN2 = BatchNormalization()

        # HG3 lower block
        self.HG3_LOW1_conv1 = SeparableConv2D(16, 3, 1, padding="same", use_bias=False)
        self.HG3_LOW1_bn1 = BatchNormalization()
        self.HG3_LOW1_conv2 = SeparableConv2D(16, 3, 1, padding="same", use_bias=False)
        self.HG3_LOW1_bn2 = BatchNormalization()

        # HG2 block (upper)
        self.HG2_BB_Upper_conv1 = SeparableConv2D(16, 3, 1, padding="same", use_bias=False)
        self.HG2_BB_Upper_bn1 = BatchNormalization()
        self.HG2_BB_Upper_conv2 = SeparableConv2D(16, 3, 1, padding="same", use_bias=False)
        self.HG2_BB_Upper_bn2 = BatchNormalization()

        # HG2 lower block
        self.HG2_LOW1_conv1 = SeparableConv2D(32, 3, 1, padding="same", use_bias=False)
        self.HG2_LOW1_bn1 = BatchNormalization()
        self.HG2_LOW1_conv2 = SeparableConv2D(32, 3, 1, padding="same", use_bias=False)
        self.HG2_LOW1_bn2 = BatchNormalization()

        # HG1 block (upper)
        self.HG1_BB_Upper_conv1 = SeparableConv2D(32, 3, 1, padding="same", use_bias=False)
        self.HG1_BB_Upper_bn1 = BatchNormalization()
        self.HG1_BB_Upper_conv2 = SeparableConv2D(32, 3, 1, padding="same", use_bias=False)
        self.HG1_BB_Upper_bn2 = BatchNormalization()

        # HG1 lower block
        self.HG1_LOW1_conv1 = SeparableConv2D(64, 3, 1, padding="same", use_bias=False)
        self.HG1_LOW1_bn1 = BatchNormalization()
        self.HG1_LOW1_conv2 = SeparableConv2D(64, 3, 1, padding="same", use_bias=False)
        self.HG1_LOW1_bn2 = BatchNormalization()

        # HG0 block (upper)
        self.HG0_BB_Upper_conv1 = SeparableConv2D(64, 3, 1, padding="same", use_bias=False)
        self.HG0_BB_Upper_bn1 = BatchNormalization()
        self.HG0_BB_Upper_conv2 = SeparableConv2D(64, 3, 1, padding="same", use_bias=False)
        self.HG0_BB_Upper_bn2 = BatchNormalization()

        # HG0 lower block
        self.HG0_LOW1_conv1 = SeparableConv2D(self.ncHG0, 3, 1, padding="same", use_bias=False)
        self.HG0_LOW1_bn1 = BatchNormalization()
        self.HG0_LOW1_conv2 = SeparableConv2D(self.ncHG0, 3, 1, padding="same", use_bias=False)
        self.HG0_LOW1_bn2 = BatchNormalization()

        # LightingNet layers on global pooled features
        self.lighting_conv1 = SeparableConv2D(128, 1, 1, use_bias=False)
        self.lighting_prelu = PReLU()
        self.lighting_conv2 = SeparableConv2D(9, 1, 1, use_bias=False)

    def call(self, inputs):
        # inputs is a list, take first element as per original code
        x = inputs[0]  # shape assumed [B, 512, 512, 1]
        feat = self.pre_conv(x)
        feat = self.pre_bn(feat)
        feat = self.relu_1(feat)

        # HG3 upper block
        feat = self.HG3_BB_Upper_conv1(feat)
        feat = self.HG3_BN1(feat)
        feat = ReLU()(feat)
        feat = self.HG3_BB_Upper_conv2(feat)
        feat = self.HG3_BN2(feat)
        feat = ReLU()(feat)

        # HG3 downsample
        feat = MaxPool2D(pool_size=2, strides=2)(feat)

        # HG3 lower block
        feat = self.HG3_LOW1_conv1(feat)
        feat = self.HG3_LOW1_bn1(feat)
        feat = ReLU()(feat)
        feat = self.HG3_LOW1_conv2(feat)
        feat = self.HG3_LOW1_bn2(feat)
        feat = ReLU()(feat)

        # HG2 upper block
        feat = self.HG2_BB_Upper_conv1(feat)
        feat = self.HG2_BB_Upper_bn1(feat)
        feat = ReLU()(feat)
        feat = self.HG2_BB_Upper_conv2(feat)
        feat = self.HG2_BB_Upper_bn2(feat)
        feat = ReLU()(feat)

        # HG2 downsample
        feat = MaxPool2D(pool_size=2, strides=2)(feat)

        # HG2 lower block
        feat = self.HG2_LOW1_conv1(feat)
        feat = self.HG2_LOW1_bn1(feat)
        feat = ReLU()(feat)
        feat = self.HG2_LOW1_conv2(feat)
        feat = self.HG2_LOW1_bn2(feat)
        feat = ReLU()(feat)

        # HG1 upper block
        feat = self.HG1_BB_Upper_conv1(feat)
        feat = self.HG1_BB_Upper_bn1(feat)
        feat = ReLU()(feat)
        feat = self.HG1_BB_Upper_conv2(feat)
        feat = self.HG1_BB_Upper_bn2(feat)
        feat = ReLU()(feat)

        # HG1 downsample
        feat = MaxPool2D(pool_size=2, strides=2)(feat)

        # HG1 lower block
        feat = self.HG1_LOW1_conv1(feat)
        feat = self.HG1_LOW1_bn1(feat)
        feat = ReLU()(feat)
        feat = self.HG1_LOW1_conv2(feat)
        feat = self.HG1_LOW1_bn2(feat)
        feat = ReLU()(feat)

        # HG0 upper block
        feat = self.HG0_BB_Upper_conv1(feat)
        feat = self.HG0_BB_Upper_bn1(feat)
        feat = ReLU()(feat)
        feat = self.HG0_BB_Upper_conv2(feat)
        feat = self.HG0_BB_Upper_bn2(feat)
        feat = ReLU()(feat)

        # HG0 downsample
        feat = MaxPool2D(pool_size=2, strides=2)(feat)

        # HG0 lower block
        feat = self.HG0_LOW1_conv1(feat)
        feat = self.HG0_LOW1_bn1(feat)
        feat = ReLU()(feat)
        feat = self.HG0_LOW1_conv2(feat)
        feat = self.HG0_LOW1_bn2(feat)
        feat = ReLU()(feat)

        # LightingNet processing:
        # Slice channels 0:27 out of the last feature map (batch, h, w, channels)
        # Take global mean spatially, output shape (batch,1,1,27)
        feat_sh = feat[:, :, :, :self.nrSH_in]
        feat_avg = tf.reduce_mean(feat_sh, axis=[1,2], keepdims=True)

        feat = self.lighting_conv1(feat_avg)
        feat = self.lighting_prelu(feat)
        L_hat = self.lighting_conv2(feat)  # output shape (batch, 1, 1, 9)

        # squeeze spatial dims to shape (batch, 9)
        out = tf.squeeze(L_hat, axis=[1,2])

        return out


def my_model_function():
    """
    Return an instance of MyModel with initialized weights.
    """
    model = MyModel()
    # Build the model by calling it once with dummy input to initialize weights
    dummy_input = GetInput()
    model(dummy_input)
    return model


def GetInput():
    """
    Return a random float32 tensor input that matches MyModel input:
    A batch of grayscale images with shape [B, 512, 512, 1].
    Here batch size B = 1 as default.
    """
    return tf.random.uniform(shape=(1, 512, 512, 1), dtype=tf.float32)

