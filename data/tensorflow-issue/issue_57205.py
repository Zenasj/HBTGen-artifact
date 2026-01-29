# tf.random.uniform((B, H, W, C), dtype=tf.float32) ← Assumed input shape is (batch, height, width, channels)
# Based on typical VGG input sizes, let's assume (1, 224, 224, 3) as a common default for ImageNet-style models.

import tensorflow as tf

class Segment(tf.keras.layers.Layer):
    def __init__(self, layers_config, **kwargs):
        super().__init__(**kwargs)
        self._layers = tf.keras.Sequential()
        # Build segment layers according to 'layers_config', mixing Conv2D+BN+ReLU and MaxPool2D
        for l in layers_config:
            if l == 'M':
                self._layers.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
            else:
                self._layers.add(tf.keras.layers.Conv2D(l, kernel_size=3, padding='same'))
                self._layers.add(tf.keras.layers.BatchNormalization())
                self._layers.add(tf.keras.layers.ReLU())
        # Add an average pooling layer at the end with pool_size=1 and stride 1 (mostly a no-op)
        self._layers.add(tf.keras.layers.AveragePooling2D(pool_size=1, strides=1))

    def call(self, x):
        # Apply tf.recompute_grad to save memory by recomputing gradients during backprop
        # This wrapper trades compute for memory by recomputing forward ops in backward pass
        return tf.recompute_grad(self._layers)(x)


class MyModel(tf.keras.Model):
    def __init__(self, vgg_name='vgg_new16', num_classes=1000):
        """
        Unified model class fusing the VGG style segments with recompute_grad,
        as described in the issue. Default to vgg_new16 config.
        """
        super().__init__()
        # VGG architectures definition similarly as in the issue
        self.config = {
            'vgg_new11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'vgg_new13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'vgg_new16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'vgg_new19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
        }
        model_cfg = self.config[vgg_name]

        self.blocks = tf.keras.Sequential()
        _seg = []
        for i, l in enumerate(model_cfg):
            _seg.append(l)
            # The original code segments the config every 3 layers into a Segment
            # This groups convolution + max pool layers in chunks of 3 elements, as the original author did.
            if (i+1) % 3 == 0:
                self.blocks.add(Segment(_seg))
                _seg = []
            elif (i+1) == len(model_cfg) and len(_seg) > 0:
                self.blocks.add(Segment(_seg))

        self.flatten = tf.keras.layers.Flatten()
        # Classification head with softmax activation for num_classes
        self.fc = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, x, **kwargs):
        # Forward pass through recompute-grad wrapped blocks,
        # flatten and final classification dense layer as typical in VGG
        out = self.blocks(x)
        out = self.flatten(out)
        out = self.fc(out)
        return out


def my_model_function():
    # Function to return an instance of MyModel with default parameters
    return MyModel()


def GetInput():
    # Return an input tensor that matches the expected VGG input shape
    # Typical VGG input size: batch 1, 224x224 RGB image, dtype float32
    return tf.random.uniform(shape=(1, 224, 224, 3), dtype=tf.float32)

