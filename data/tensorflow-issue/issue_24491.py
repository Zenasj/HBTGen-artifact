# tf.random.uniform((B, H, W, C), dtype=tf.float32)
import tensorflow as tf

class MyModel(tf.keras.Model):
    """
    A fused model that simulates ResNet-like behavior converted from PyTorch via ONNX to TensorFlow,
    with explicit handling/conversion of input data format from NCHW to NHWC, to make it compatible with TFLite.
    
    Since the issue revolves around incompatibility with TensorFlow Lite conversion due to data_format='NCHW',
    this model internally supports inputs in NCHW format (PyTorch style),
    converts them to NHWC format (TensorFlow Lite required), then forwards through a basic
    convolutional stack resembling part of a typical ResNet stem. This emulates the typical
    dimension transposition fix hinted in the issue.
    
    The forward pass returns the output tensor after the NHWC conversion and convolution.
    
    This model structure is simplified and reasonable given the lack of full original code,
    with assumptions:
    - Input tensor shape is (B, C, H, W), common PyTorch format.
    - Use a transpose layer to convert to NHWC before applying Conv2D, BatchNorm, ReLU.
    """

    def __init__(self):
        super().__init__()
        # A minimal conv block resembling part of ResNet initial layers
        self.conv1 = tf.keras.layers.Conv2D(
            filters=64, kernel_size=7, strides=2, padding='same', use_bias=False,
            name='conv1')
        self.bn1 = tf.keras.layers.BatchNormalization(name='bn1')
        self.relu = tf.keras.layers.ReLU(name='relu')

    def call(self, inputs, training=False):
        # inputs shape: (B, C, H, W) - NCHW as from PyTorch/ONNX model
        # Transpose to NHWC: (B, H, W, C)
        x = tf.transpose(inputs, perm=[0, 2, 3, 1])

        # Forward through convolutional block (NHWC)
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.relu(x)

        return x


def my_model_function():
    """
    Returns an instance of MyModel.
    Assumes default initialization; no external weights loaded.
    """
    return MyModel()


def GetInput():
    """
    Generates a random input tensor matching the input expected by MyModel.
    The input is assumed to be NCHW as from PyTorch models before conversion.
    
    Assumptions:
    - Batch size = 1
    - Channels = 3 (RGB image)
    - Height = 224
    - Width = 224
    
    These are standard dimensions for many ImageNet-pretrained models like ResNet.
    dtype=tf.float32 to match typical TensorFlow model input.
    """
    B, C, H, W = 1, 3, 224, 224
    return tf.random.uniform((B, C, H, W), dtype=tf.float32)

