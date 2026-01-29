# tf.random.uniform((1, 28, 28, 1), dtype=tf.float32) ← inferred input shape from MNIST example

import tensorflow as tf
from tensorflow.keras import layers, activations
import numpy as np

class C3BR(tf.keras.Model):
    ''' 3D Convolution + Batch Normalisation + Relu '''
    def __init__(self, filterNum, kSize, strSize, padMode):
        super(C3BR, self).__init__()
        # Conv3D with channels_first to match given design
        self.conv = layers.Conv3D(filters=filterNum, kernel_size=kSize, 
                                  strides=strSize, padding=padMode, 
                                  data_format='channels_first')
        self.BN = layers.BatchNormalization(axis=1)
    
    def call(self, inputs, ifTrain=True):
        x = self.conv(inputs)
        if ifTrain:
            x = self.BN(x, training=ifTrain)  # pass training flag to BN
        return activations.relu(x)

class SimpleUNet(tf.keras.Model):
    """
    A double-layered encoder-decoder U-Net model for 3D inputs, with channels_first format.
    Input:
        inDim: tuple, e.g. (channels, depth, height, width)
        classNum: int, number of classes including background
    Output:
        Tensor of shape [batch_size, classNum, D', H', W'] with softmax probabilities
    Notes:
        Shapes are carefully cropped to match dimensions after downsampling and upsampling.
        Use 'ifTrain' flag to toggle batchNorm behavior.
    """
    def __init__(self, inDim, classNum, name='SimpleUNet', **kwargs):
        super(SimpleUNet, self).__init__(name=name, **kwargs)
        self.inDim = inDim
        self.classNum = classNum

        # Compute shape after layers for cropping -- as per the original user code logic
        # Convert input dims to numpy array for math: ignoring batch dim, so shape is (channels, D, H, W)
        dims = np.array(inDim[1:], dtype=np.int32)
        # Calculated using original formulas in chunks, approximate integer operations:
        dimEnSt1End = dims - 2 - 2  # after first two conv3d layers with kernel=3 and 'valid'
        dimEnSt2Ed = (dimEnSt1End // 2) - 2 - 2  # after maxpool stride2 + two conv3d valids
        dimBridgeEnd = ((dimEnSt2Ed // 2) - 2 - 2) * 2  # after maxpool, two conv3d, and transpose conv doubling spatial size
        dimDEStd1End = (dimBridgeEnd - 2 - 2) * 2  # after cropping, two conv3d and one transpose conv doubling size

        self.outDim = dimDEStd1End - 2 - 2 - 2  # final cropping and conv3d

        # Cropping tuples, amount to crop on each side for concat skip connections
        crop3d1_amount = ((dimEnSt2Ed - dimBridgeEnd) // 2,)
        crop3d1 = ((crop3d1_amount[0], crop3d1_amount[0]),) * 3  # for D, H, W
        crop3d2_amount = ((dimEnSt1End - dimDEStd1End) // 2,)
        crop3d2 = ((crop3d2_amount[0], crop3d2_amount[0]),) * 3

        # Encoder stage 1
        self.en_st1_cbr1 = C3BR(32, 3, 1, 'valid')
        self.en_st1_cbr2 = C3BR(64, 3, 1, 'valid')
        # Encoder stage 2
        self.en_st2_mp = layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), 
                                             padding='valid', data_format='channels_first')
        self.en_st2_cbr1 = C3BR(64, 3, 1, 'valid')
        self.en_st2_cbr2 = C3BR(128, 3, 1, 'valid')
        # Bridge
        self.bridge_mp = layers.MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), 
                                             padding='valid', data_format='channels_first')
        self.bridge_cbr1 = C3BR(128, 3, 1, 'valid')
        self.bridge_cbr2 = C3BR(256, 3, 1, 'valid')    
        self.bridge_tconv1 = layers.Conv3DTranspose(256, 2, strides=2, padding='valid',
                                                    data_format='channels_first')
        # Decoder
        self.de_3dcrop1 = layers.Cropping3D(cropping=crop3d1, data_format='channels_first')
        self.de_st1_cbr1 = C3BR(256, 3, 1, 'valid')
        self.de_st1_cbr2 = C3BR(128, 3, 1, 'valid')    
        self.de_st1_tconv1 = layers.Conv3DTranspose(128, 2, strides=2, padding='valid', 
                                                    data_format='channels_first')
        self.de_3dcrop2 = layers.Cropping3D(cropping=crop3d2, data_format='channels_first')
        self.de_st2_cbr1 = C3BR(64, 3, 1, 'valid')
        self.de_st2_cbr2 = C3BR(64, 3, 1, 'valid') 
        self.final_conv3D = layers.Conv3D(filters=self.classNum, kernel_size=3, strides=1, padding='valid', 
                                         data_format='channels_first')
                
    def call(self, inputs, ifTrain=True):
        x = self.en_st1_cbr1(inputs, ifTrain)
        xEnSt1End = self.en_st1_cbr2(x, ifTrain)
        x = self.en_st2_mp(xEnSt1End)
        x = self.en_st2_cbr1(x, ifTrain)
        xEnSt2Ed = self.en_st2_cbr2(x, ifTrain)
        x = self.bridge_mp(xEnSt2Ed)        
        x = self.bridge_cbr1(x, ifTrain)
        x = self.bridge_cbr2(x, ifTrain)      
        xBridgeEnd = self.bridge_tconv1(x)
        xCrop1 = self.de_3dcrop1(xEnSt2Ed)

        # Concatenate along channel axis=1 (channels_first)
        x = layers.concatenate([xBridgeEnd, xCrop1], axis=1)

        x = self.de_st1_cbr1(x, ifTrain)
        x = self.de_st1_cbr2(x, ifTrain)
        xDeSt1End = self.de_st1_tconv1(x)
        xCrop2 = self.de_3dcrop2(xEnSt1End)
        x = layers.concatenate([xDeSt1End, xCrop2], axis=1)

        x = self.de_st2_cbr1(x, ifTrain)
        x = self.de_st2_cbr2(x, ifTrain)
        x = self.final_conv3D(x)
        outputs = activations.softmax(x, axis=1)
        return outputs
        
    def compute_output_shape(self, input_shape):
        # Compute output shape for model compatibility
        batch_size = input_shape[0]
        return (batch_size, self.classNum) + tuple(self.outDim.tolist())


class CNN(tf.keras.Model):
    """Simple CNN model from MNIST example in TensorFlow eager mode issue."""
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.cnn1 = layers.Conv2D(16, (5, 5), padding='same', strides=(2, 2))
        self.bn1 = layers.BatchNormalization()
        self.cnn2 = layers.Conv2D(32, (5, 5), padding='same', strides=(2, 2))
        self.bn2 = layers.BatchNormalization()
        self.pool = layers.GlobalAveragePooling2D()
        self.classifier = layers.Dense(num_classes)

    def call(self, inputs, training=False):
        # The original model prints input shape to show problematic dataset pass
        # We keep print for demonstration/logging purposes as user did
        tf.print("CNN call input shape:", tf.shape(inputs))
        x = self.cnn1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.cnn2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool(x)
        logits = self.classifier(x)
        # Use CPU for softmax as per original note (though newer TF supports GPU softmax)
        output = tf.nn.softmax(logits)
        return output


class MyModel(tf.keras.Model):
    """
    Fused wrapper model containing both the 2D CNN and 3D SimpleUNet sub-models.
    This model can run either and compare their outputs or return either, depending on context.
    Here, the forward method returns a dict of their outputs.

    Since the original issue discussed two different models (CNN on 4D images, SimpleUNet on 5D tensors),
    input handling requires separate inputs.

    For demonstration, forward signatures:
       - inputs: dict with keys 'cnn_input' and 'unet_input'
       - outputs: dict with keys 'cnn_output' and 'unet_output'

    This fusion is to represent "multiple models discussed together"
    """

    def __init__(self):
        super(MyModel, self).__init__()
        # CNN for 2D MNIST style data, input shape [batch, 28, 28, 1]
        self.cnn = CNN(num_classes=10)
        # SimpleUNet expects inputs with channels_first, 5D: [batch, channels, D, H, W]
        # We'll use inDim and classNum inferred from chunks above:
        self.unet = SimpleUNet(inDim=(4, 64, 64, 64), classNum=2)

    def call(self, inputs, training=False):
        """
        inputs: dict with:
          'cnn_input': tensor of shape (B, 28, 28, 1)
          'unet_input': tensor of shape (B, 4, 64, 64, 64), channels_first 3D volumes

        Returns:
          dict with keys 'cnn_output', 'unet_output'
        """
        cnn_inp = inputs['cnn_input']
        unet_inp = inputs['unet_input']

        cnn_out = self.cnn(cnn_inp, training=training)
        unet_out = self.unet(unet_inp, ifTrain=training)

        return {'cnn_output': cnn_out, 'unet_output': unet_out}


def my_model_function():
    """
    Returns:
        Instance of MyModel, with sub-models initialized.
    """
    return MyModel()


def GetInput():
    """
    Returns:
        A dictionary of inputs to feed into MyModel.call(), matching their expected shapes and dtypes.
    """
    # CNN input: batch size 1, 28x28 grayscale image normalized [0,1]
    cnn_input = tf.random.uniform((1, 28, 28, 1), dtype=tf.float32)

    # UNet input: batch size 1, 4 channels, 64x64x64 volume 
    # (channels_first 5D tensor as per SimpleUNet expectations)
    unet_input = tf.random.uniform((1, 4, 64, 64, 64), dtype=tf.float32)

    return {'cnn_input': cnn_input, 'unet_input': unet_input}

