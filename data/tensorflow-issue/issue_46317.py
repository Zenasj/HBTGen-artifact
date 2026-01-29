# tf.random.uniform((1, 224, 224, 3), dtype=tf.float32) ← the example input shape from the issue code

import tensorflow as tf
from tensorflow.keras import layers, Sequential

class ConvBNReLU(layers.Layer):
    def __init__(self, out_channel, kernel_size=3, stride=1, **kwargs):
        super(ConvBNReLU, self).__init__(**kwargs)
        # Using explicit layer names for stable weight names across TF versions:
        # The main issue discussed is name stability of weights created by Sequential and layers.
        # To improve reproducibility, we assign unique explicit names for each layer inside this block,
        # which may reduce the name flattening issue reported for TF 2.3, 2.4 and later.
        self.conv_name = kwargs.get("conv_name", "Conv2d")
        self.bn_name = kwargs.get("bn_name", "BatchNorm")

        self.combine_layer = Sequential([
            layers.Conv2D(filters=out_channel, kernel_size=kernel_size,
                          strides=stride, padding='SAME', use_bias=False, name=self.conv_name),
            layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=self.bn_name),
            layers.ReLU(max_value=6.0)
        ], name="combine")

    def call(self, inputs, training=False, **kwargs):
        x = self.combine_layer(inputs, training=training)
        return x

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        # To get distinct variable names as in TF 2.0~2.2 style, we give unique layer names here.
        # This attempts to simulate the naming style where submodules names are preserved
        # instead of being flattened as just the inner layer name.
        self.convbnrelu1 = ConvBNReLU(32, stride=2, conv_name="Conv2d1", bn_name="BatchNorm1")
        self.convbnrelu2 = ConvBNReLU(64, stride=2, conv_name="Conv2d2", bn_name="BatchNorm2")

    def call(self, inputs, training=False, **kwargs):
        x = self.convbnrelu1(inputs, training=training)
        x = self.convbnrelu2(x, training=training)
        return x

def my_model_function():
    # Return an instance of MyModel properly initialized.
    # No pretrained weights loaded here as the issue relates to naming and weight loading.
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel.
    # Input shape: batch=1 (reasonable assumption), 224x224 image with 3 channels as per the issue example.
    return tf.random.uniform((1, 224, 224, 3), dtype=tf.float32)

