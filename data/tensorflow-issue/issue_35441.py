# tf.random.uniform((2, 4, 64, 64, 64), dtype=tf.float32) ← Assumed input shape from issue (mbSize=2, channels=4, 64³ volume), channels_first

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, activations


class MyModel(tf.keras.Model):
    """Combined model from original issue's SimpleUNet1 with C3BR blocks.

    This is a 3D U-Net style segmentation model working on
    5D tensors: [batch, channels (modality), depth, height, width]
    with channel-first data format.

    The model uses explicit Conv3D + BatchNorm + ReLU blocks chained,
    max-pooling and transposed convolution for down- and up-sampling,
    and concatenation along channels axis. Cropping is applied in the decoder
    to match spatial shapes before concatenation.

    Note: The original issue distinguished behavior difference between using
    layers.Concatenate() layer vs functional layers.concatenate(). Here,
    we follow the working approach with layers.Concatenate instances
    to ensure model saving stability.

    The call() method supports a train-mode flag to activate batch norm.
    """

    def __init__(self, inDim, classNum, name="MyModel", **kwargs):
        super().__init__(name=name, **kwargs)
        self.inDim = inDim  # Input dimension tuple, e.g. (channels, dim1, dim2, dim3)
        self.classNum = classNum

        # Following input dimension calculations from original code:
        # numpy array of spatial dims (excluding batch and channels)
        spatial_dims = np.array(inDim[1:])

        dimEnSt1End = spatial_dims - 2 - 2
        dimEnSt2Ed = dimEnSt1End / 2 - 2 - 2
        dimBridgeEnd = (dimEnSt2Ed / 2 - 2 - 2) * 2
        dimDEStd1End = (dimBridgeEnd - 2 - 2) * 2
        self.outDim = (dimDEStd1End - 2 - 2 - 2).astype(int)

        # Calculate cropping for Cropping3D layers as tuples of (before, after)
        temp1 = ((dimEnSt2Ed - dimBridgeEnd) / 2).astype("int32")
        crop3d1 = tuple(np.tile(temp1, (2, 1)).T)
        temp2 = ((dimEnSt1End - dimDEStd1End) / 2).astype("int32")
        crop3d2 = tuple(np.tile(temp2, (2, 1)).T)

        # Define the convolutional blocks
        self.en_st1_cbr1 = C3BR(32, 3, 1, "valid")
        self.en_st1_cbr2 = C3BR(64, 3, 1, "valid")

        self.en_st2_mp = layers.MaxPooling3D(
            pool_size=(2, 2, 2),
            strides=(2, 2, 2),
            padding="valid",
            data_format="channels_first",
        )
        self.en_st2_cbr1 = C3BR(64, 3, 1, "valid")
        self.en_st2_cbr2 = C3BR(128, 3, 1, "valid")

        self.bridge_mp = layers.MaxPooling3D(
            pool_size=(2, 2, 2),
            strides=(2, 2, 2),
            padding="valid",
            data_format="channels_first",
        )
        self.bridge_cbr1 = C3BR(128, 3, 1, "valid")
        self.bridge_cbr2 = C3BR(256, 3, 1, "valid")

        self.bridge_tconv1 = layers.Conv3DTranspose(
            256, 2, strides=2, padding="valid", data_format="channels_first"
        )

        self.de_3dcrop1 = layers.Cropping3D(cropping=crop3d1, data_format="channels_first")
        self.de_st1_concat = layers.Concatenate(axis=1)  # channel axis

        self.de_st1_cbr1 = C3BR(256, 3, 1, "valid")
        self.de_st1_cbr2 = C3BR(128, 3, 1, "valid")

        self.de_st1_tconv1 = layers.Conv3DTranspose(
            128, 2, strides=2, padding="valid", data_format="channels_first"
        )
        self.de_3dcrop2 = layers.Cropping3D(cropping=crop3d2, data_format="channels_first")
        self.de_st2_concat = layers.Concatenate(axis=1)

        self.de_st2_cbr1 = C3BR(64, 3, 1, "valid")
        self.de_st2_cbr2 = C3BR(64, 3, 1, "valid")

        self.final_conv3D = layers.Conv3D(
            filters=self.classNum, kernel_size=3, strides=1, padding="valid", data_format="channels_first"
        )

    def call(self, inputs, ifTrain=True):
        # Encoder stage 1
        x0 = self.en_st1_cbr1(inputs, ifTrain=ifTrain)
        xEnSt1End = self.en_st1_cbr2(x0, ifTrain=ifTrain)

        # Encoder stage 2 - pooling + conv
        x1 = self.en_st2_mp(xEnSt1End)
        x2 = self.en_st2_cbr1(x1, ifTrain=ifTrain)
        xEnSt2Ed = self.en_st2_cbr2(x2, ifTrain=ifTrain)

        # Bridge
        x3 = self.bridge_mp(xEnSt2Ed)
        x4 = self.bridge_cbr1(x3, ifTrain=ifTrain)
        x5 = self.bridge_cbr2(x4, ifTrain=ifTrain)

        xBridgeEnd = self.bridge_tconv1(x5)
        xCrop1 = self.de_3dcrop1(xEnSt2Ed)

        x6 = self.de_st1_concat([xBridgeEnd, xCrop1])
        x7 = self.de_st1_cbr1(x6, ifTrain=ifTrain)
        x8 = self.de_st1_cbr2(x7, ifTrain=ifTrain)

        xDeSt1End = self.de_st1_tconv1(x8)
        xCrop2 = self.de_3dcrop2(xEnSt1End)

        x9 = self.de_st2_concat([xDeSt1End, xCrop2])
        x10 = self.de_st2_cbr1(x9, ifTrain=ifTrain)
        x11 = self.de_st2_cbr2(x10, ifTrain=ifTrain)

        x12 = self.final_conv3D(x11)

        outputs = activations.softmax(x12, axis=1)  # softmax over channel dimension

        return outputs

    def build_model(self, input_shape):
        # Work-around to define shapes by running dummy inputs through model
        self.build(input_shape)
        inputs = tf.keras.Input(shape=input_shape[1:])
        _ = self.call(inputs)

    def compute_output_shape(self):
        # Optional override for Keras fit compatibility (not always needed)
        return tf.TensorShape(np.append(self.classNum, self.outDim))


class C3BR(tf.keras.Model):
    """3D Conv + BatchNorm + ReLU block (channels_first)."""

    def __init__(self, filterNum, kSize, strSize, padMode):
        super().__init__()
        self.conv = layers.Conv3D(
            filters=filterNum,
            kernel_size=kSize,
            strides=strSize,
            padding=padMode,
            data_format="channels_first",
        )
        self.BN = layers.BatchNormalization(axis=1)

    def call(self, inputs, ifTrain=True):
        x = self.conv(inputs)
        if ifTrain:
            x = self.BN(x)
        return activations.relu(x)

    def build_model(self, input_shape):
        self.build(input_shape)
        inputs = tf.keras.Input(shape=input_shape[1:])
        _ = self.call(inputs)


def my_model_function():
    model_in_dim = (4, 64, 64, 64)  # channels=4, spatial dims 64³
    class_num = 2
    return MyModel(model_in_dim, class_num)


def GetInput():
    # Generate a random input tensor with batch size 2 and same input dims/channels
    mbSize = 2
    channels = 4
    depth = 64
    height = 64
    width = 64
    # Use float32 as default dtype, channels_first format, shape 5D: [batch, channels, D, H, W]
    return tf.random.uniform((mbSize, channels, depth, height, width), dtype=tf.float32)

