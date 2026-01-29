# tf.random.uniform((B, C, H, W, D), dtype=tf.float32) ← inferred input shape (batch_size, channels, depth, height, width)
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, activations

class CBR(layers.Layer):
    """
    Convolution + Batch Normalization + ReLU
    This is a fixed version of the original CBR custom layer.
    The original had __int__ typo instead of __init__ causing the error.
    """
    def __init__(self, filterNum, kSize, strSize, padMode, name='cbr', **kwargs):
        super(CBR, self).__init__(name=name, **kwargs)
        self.conv3D = layers.Conv3D(
            filters=filterNum, kernel_size=kSize, strides=strSize,
            padding=padMode, data_format='channels_first')
        self.BN = layers.BatchNormalization(axis=1)

    def call(self, inputs):
        x = self.conv3D(inputs)
        x = self.BN(x)
        return activations.relu(x)


class MyModel(tf.keras.Model):
    """
    A custom SimpleUNet model built from basic units including CBR, max pooling,
    conv3d transpose upsampling, cropping and concatenation.
    
    Inputs:
      inDim: [channels, depth, height, width] assuming fixed batch size at runtime
      classNum: number of output classes/channels
    
    The original code computes various cropping amounts based on input dims
    and chaining Conv3D with 'valid' padding leading to spatial dimensionality shrinking.

    Shapes are assumed channels_first: [batch, channels, depth, height, width]
    """
    def __init__(self, inDim=(4, 64, 64, 64), classNum=3, name='SimpleUNet', **kwargs):
        super(MyModel, self).__init__(name=name, **kwargs)

        # Save attributes
        self.inDim = inDim
        self.classNum = classNum

        # Convert input spatial dims to numpy array for cropping calcs
        spatial_dims = np.array(inDim[1:])

        # Compute cropping sizes along each spatial axis based on chain of valid padding convolutions/poolings
        dimEnSt1End = spatial_dims - 2 - 2    # two conv3d with kernel_size=3 reduce each dim by 2 each (padding='valid')
        dimEnSt2Ed = dimEnSt1End / 2 - 2 - 2  # pooling halves, two conv3d reduce by 4 more
        dimBridgeEnd = (dimEnSt2Ed / 2 - 2 - 2) * 2  # conv3ds and deconv eventually double back
        dimDEStd1End = (dimBridgeEnd - 2 - 2) * 2
        outDim = dimDEStd1End - 2 - 2 - 2

        # Calculate cropping tuples for Cropping3D layers, must be integers
        temp1 = ((dimEnSt2Ed - dimBridgeEnd) / 2).astype('int32')
        crop3d1 = tuple(np.tile(temp1, (2, 1)).T)

        temp2 = ((dimEnSt1End - dimDEStd1End) / 2).astype('int32')
        crop3d2 = tuple(np.tile(temp2, (2, 1)).T)

        # Encoder stage 1
        self.en_st1_cbr1 = CBR(32, 3, 1, 'valid')
        self.en_st1_cbr2 = CBR(64, 3, 1, 'valid')

        # Encoder stage 2
        self.en_st2_mp = layers.MaxPooling3D(
            pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid',
            data_format='channels_first')
        self.en_st2_cbr1 = CBR(64, 3, 1, 'valid')
        self.en_st2_cbr2 = CBR(128, 3, 1, 'valid')

        # Bridge
        self.bridge_mp = layers.MaxPooling3D(
            pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid',
            data_format='channels_first')
        self.bridge_cbr1 = CBR(128, 3, 1, 'valid')
        self.bridge_cbr2 = CBR(256, 3, 1, 'valid')
        self.bridge_tconv1 = layers.Conv3DTranspose(
            256, 2, strides=2, padding='valid', data_format='channels_first')

        # Decoder stage 1
        self.de_3dcrop1 = layers.Cropping3D(cropping=crop3d1, data_format='channels_first')
        self.de_st1_cbr1 = CBR(256, 3, 1, 'valid')
        self.de_st1_cbr2 = CBR(128, 3, 1, 'valid')
        self.de_st1_tconv1 = layers.Conv3DTranspose(
            128, 2, strides=2, padding='valid', data_format='channels_first')

        # Decoder stage 2
        self.de_3dcrop2 = layers.Cropping3D(cropping=crop3d2, data_format='channels_first')
        self.de_st2_cbr1 = CBR(64, 3, 1, 'valid')
        self.de_st2_cbr2 = CBR(64, 3, 1, 'valid')

        # Final convolution to produce output classes
        self.final_conv3D = layers.Conv3D(
            filters=self.classNum, kernel_size=3, strides=1,
            padding='valid', data_format='channels_first')

    def call(self, inputs):
        # Encoder stage 1
        x = self.en_st1_cbr1(inputs)
        xEnSt1End = self.en_st1_cbr2(x)

        # Encoder stage 2
        x = self.en_st2_mp(xEnSt1End)
        x = self.en_st2_cbr1(x)
        xEnSt2Ed = self.en_st2_cbr2(x)

        # Bridge
        x = self.bridge_mp(xEnSt2Ed)
        x = self.bridge_cbr1(x)
        x = self.bridge_cbr2(x)
        xBridgeEnd = self.bridge_tconv1(x)

        # Crop and concatenate skip connection from encoder stage 2
        xCrop1 = self.de_3dcrop1(xEnSt2Ed)
        x = layers.Concatenate(axis=1)([xBridgeEnd, xCrop1])

        # Decoder stage 1 conv blocks
        x = self.de_st1_cbr1(x)
        x = self.de_st1_cbr2(x)
        xDeSt1End = self.de_st1_tconv1(x)

        # Crop and concatenate skip connection from encoder stage 1
        xCrop2 = self.de_3dcrop2(xEnSt1End)
        x = layers.Concatenate(axis=1)([xDeSt1End, xCrop2])

        # Decoder stage 2 conv blocks
        x = self.de_st2_cbr1(x)
        x = self.de_st2_cbr2(x)

        # Final conv layer
        x = self.final_conv3D(x)

        # Softmax activation along class dimension (channel axis = 1)
        outputs = activations.softmax(x, axis=1)

        return outputs


def my_model_function():
    # Return an instance of the SimpleUNet (MyModel) with default input and 3 classes
    # These defaults match the original problem example in chunks
    inDim = (4, 64, 64, 64)  # (channels, depth, height, width)
    classNum = 3
    return MyModel(inDim=inDim, classNum=classNum)

def GetInput():
    # Return a random tensor input matching MyModel expected input shape
    # Shape: (batch_size, channels, depth, height, width)
    batch_size = 1  # batch size can be 1 for testing or more
    inDim = (4, 64, 64, 64)
    input_shape = (batch_size,) + inDim
    return tf.random.uniform(input_shape, dtype=tf.float32)

