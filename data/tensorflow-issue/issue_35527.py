# tf.random.uniform((1, 4, 64, 64, 64), dtype=tf.float32) ← Input shape: batch size 1, 4 channels/modalities, 64x64x64 volume

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, activations

class C3BR(tf.keras.Model):
    '''3D Convolution + Batch Normalisation + ReLU'''
    def __init__(self, filterNum, kSize, strSize, padMode):
        super(MyModel, self).__init__()
        self.conv = layers.Conv3D(
            filters=filterNum,
            kernel_size=kSize,
            strides=strSize,
            padding=padMode,
            data_format='channels_first')
        self.BN = layers.BatchNormalization(axis=1)
    
    def call(self, inputs, ifTrain=False):
        x = self.conv(inputs)
        if ifTrain:
            x = self.BN(x)
        return activations.relu(x)

class SimpleUNet(tf.keras.Model):
    """
    Basic 3D U-Net with double layered encoder-decoder structure using C3BR blocks.
    Input shape: (batch, channels=modalities, depth, height, width) with channels_first ordering.
    Output shape: (batch, classNum, reduced spatial dims)
    """
    def __init__(self, inDim, classNum):
        super(MyModel, self).__init__()
        # inDim: tuple (channels, D, H, W)
        self.inDim = inDim
        self.classNum = classNum

        # Calculate output spatial dimensions at various stages based on valid padding conv and pooling
        # Using int arithmetic carefully for cropping and concatenation calculations
        # This was inferred from original numpy code adjusted for integer casting
        dimEnSt1End = np.array(inDim[1:]) - 2 - 2  # two valid conv3d with kernel=3 reduce dims by 2 each
        dimEnSt2Ed = (dimEnSt1End // 2) - 2 - 2   # max pooling halves dims, then two conv3d reduce by 2 each
        dimBridgeEnd = ((dimEnSt2Ed // 2) - 2 - 2) * 2  # bridge layer conv and transpose conv double dims back
        dimDEStd1End = (dimBridgeEnd - 2 - 2) * 2
        self.outDim = dimDEStd1End - 2 - 2 - 2

        # Calculate cropping sizes for skip connections
        temp = ((dimEnSt2Ed - dimBridgeEnd) // 2).astype('int32')
        crop3d1 = tuple(np.tile(temp, (2, 1)).T)  # ((crop_left, crop_right), ...)
        temp = ((dimEnSt1End - dimDEStd1End) // 2).astype('int32')
        crop3d2 = tuple(np.tile(temp, (2, 1)).T)

        # Encoder stage 1
        self.en_st1_cbr1 = C3BR(32, 3, 1, 'valid')
        self.en_st1_cbr2 = C3BR(64, 3, 1, 'valid')

        # Encoder stage 2
        self.en_st2_mp = layers.MaxPooling3D(
            pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', data_format='channels_first')
        self.en_st2_cbr1 = C3BR(128, 3, 1, 'valid')
        self.en_st2_cbr2 = C3BR(128, 3, 1, 'valid')

        # Bridge
        self.bridge_mp = layers.MaxPooling3D(
            pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', data_format='channels_first')
        self.bridge_cbr1 = C3BR(256, 3, 1, 'valid')
        self.bridge_cbr2 = C3BR(256, 3, 1, 'valid')
        self.bridge_tconv1 = layers.Conv3DTranspose(
            512, 2, strides=2, padding='valid', data_format='channels_first')

        # Decoder stage 1
        self.de_3dcrop1 = layers.Cropping3D(cropping=crop3d1, data_format='channels_first')
        self.de_st1_concat = layers.Concatenate(axis=1)
        self.de_st1_cbr1 = C3BR(256, 3, 1, 'valid')
        self.de_st1_cbr2 = C3BR(128, 3, 1, 'valid')
        self.de_st1_tconv1 = layers.Conv3DTranspose(
            128, 2, strides=2, padding='valid', data_format='channels_first')

        # Decoder stage 2
        self.de_3dcrop2 = layers.Cropping3D(cropping=crop3d2, data_format='channels_first')
        self.de_st2_concat = layers.Concatenate(axis=1)
        self.de_st2_cbr1 = C3BR(64, 3, 1, 'valid')
        self.de_st2_cbr2 = C3BR(64, 3, 1, 'valid')

        # Final conv layer to classNum outputs
        self.final_conv3D = layers.Conv3D(
            filters=self.classNum, kernel_size=3, strides=1, padding='valid', data_format='channels_first')

    def call(self, inputs, ifTrain=False):
        x0 = self.en_st1_cbr1(inputs, ifTrain)
        xEnSt1End = self.en_st1_cbr2(x0, ifTrain)
        x1 = self.en_st2_mp(xEnSt1End)
        x2 = self.en_st2_cbr1(x1, ifTrain)
        xEnSt2Ed = self.en_st2_cbr2(x2, ifTrain)
        x3 = self.bridge_mp(xEnSt2Ed)
        x4 = self.bridge_cbr1(x3, ifTrain)
        x5 = self.bridge_cbr2(x4, ifTrain)
        xBridgeEnd = self.bridge_tconv1(x5)

        xCrop1 = self.de_3dcrop1(xEnSt2Ed)
        x6 = self.de_st1_concat([xBridgeEnd, xCrop1])
        x7 = self.de_st1_cbr1(x6, ifTrain)
        x8 = self.de_st1_cbr2(x7, ifTrain)
        xDeSt1End = self.de_st1_tconv1(x8)

        xCrop2 = self.de_3dcrop2(xEnSt1End)
        x9 = self.de_st2_concat([xDeSt1End, xCrop2])
        x10 = self.de_st2_cbr1(x9, ifTrain)
        x11 = self.de_st2_cbr2(x10, ifTrain)
        x12 = self.final_conv3D(x11)

        outputs = tf.cast(activations.softmax(x12, axis=1), tf.float32)
        return outputs

    def build_model(self, input_shape):
        # Helper to build model weights and see output shapes
        self.build(input_shape)
        inputs = tf.keras.Input(shape=input_shape[1:])
        _ = self.call(inputs)

    def compute_output_shape(self):
        # Return output shape as [batch, classNum, spatial dims]
        return tf.TensorShape((self.classNum,) + tuple(self.outDim))

class MyModel(tf.keras.Model):
    """
    Encapsulate the SimpleUNet subclassed model, replicating the code example context.
    """

    def __init__(self):
        super(MyModel, self).__init__()
        # Instantiating a SimpleUNet for the given example input/output size
        self.simple_unet = SimpleUNet(inDim=(4, 64, 64, 64), classNum=2)
        # Build model weights with example input shape: batch=1
        self.simple_unet.build_model(input_shape=(1, 4, 64, 64, 64))

    def call(self, inputs, ifTrain=False):
        # Forward call to the SimpleUNet
        return self.simple_unet(inputs, ifTrain=ifTrain)

def my_model_function():
    # Return an instance of MyModel with SimpleUNet initialized
    return MyModel()

def GetInput():
    # Return a random tensor input matching the input expected by MyModel
    # Shape: (batch=1, 4 modalities, 64x64x64 volume), dtype float32
    return tf.random.uniform(shape=(1, 4, 64, 64, 64), dtype=tf.float32)

