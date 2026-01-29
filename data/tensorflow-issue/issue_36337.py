# tf.random.uniform((32, 300, 300, 3), dtype=tf.float32)

import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, ReLU, MaxPool2D, ZeroPadding2D
from tensorflow.keras import Model, Sequential

feature_extractor_config = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
}

feature_pyramid_config = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
}

head_config = {
    '300': [4, 6, 6, 6, 4, 4]
}


def create_feature_extractor(cfg, batch_norm=False):
    """
    Builds the VGG-like feature extractor layers for SSD based on a config list.
    """
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [MaxPool2D(pool_size=2, strides=2)]
        elif v == 'C':
            # 'C' denotes a special kind of maxpool with asymmetric padding
            layers += [ZeroPadding2D(padding=((0, 1), (0, 1))), MaxPool2D(pool_size=2, strides=2)]
        else:
            # Manually add padding of 1px before conv (PyTorch-like padding)
            pad = ZeroPadding2D(1)
            conv2d = Conv2D(v, kernel_size=3)
            if batch_norm:
                layers += [pad, conv2d, BatchNormalization(v), ReLU()]
            else:
                layers += [pad, conv2d, ReLU()]
            in_channels = v
    # Following SSD paper: pool5 with stride 1, dilation conv6, conv7 layers
    pad5 = ZeroPadding2D(1)
    pool5 = MaxPool2D(pool_size=3, strides=1, padding='same')  # Add padding='same' to avoid size reduction
    pad6 = ZeroPadding2D(6)
    conv6 = Conv2D(1024, kernel_size=3, dilation_rate=6)
    conv7 = Conv2D(1024, kernel_size=1)
    layers += [pad5, pool5, pad6, conv6, ReLU(), conv7, ReLU()]
    return layers


def create_feature_pyramid(cfg, size=300):
    """
    Builds additional feature pyramid layers after VGG base for extra SSD detection layers.
    """
    layers = []
    in_channels = None
    flag = False
    for k, v in enumerate(cfg):
        if k == 0 or in_channels != 'S':
            if v == 'S':
                # 'S' denotes stride 2 conv with padding 1
                layers += [ZeroPadding2D(padding=1),
                           Conv2D(cfg[k + 1], kernel_size=(1, 3)[flag], strides=2)]
            else:
                layers += [Conv2D(v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    if size == 512:
        layers.append(Conv2D(128, kernel_size=1, strides=1))
        layers.append(ZeroPadding2D(padding=1))
        layers.append(Conv2D(256, kernel_size=4, strides=1))
    return layers


def create_head(cfg, num_classes):
    """
    Creates classification and regression heads consisting of Conv2D layers in Sequential containers.
    """
    reg_layers = []
    cls_layers = []
    for num_bboxes in cfg:
        cls_layers.append(Sequential([
            ZeroPadding2D(padding=1),
            Conv2D(num_bboxes * num_classes, kernel_size=3)
        ]))
        reg_layers.append(Sequential([
            ZeroPadding2D(padding=1),
            Conv2D(num_bboxes * 4, kernel_size=3)
        ]))
    head = {'reg': reg_layers, 'cls': cls_layers}
    return head


class L2Norm(Layer):
    """
    Implements L2 normalization on the feature maps with a learnable scaling factor.
    """

    def __init__(self, in_channels, scale=None):
        super(L2Norm, self).__init__()
        self.in_channels = in_channels
        # Learnable scale initialized as constant 20 per channel
        self.w = self.add_weight(
            name='w',
            shape=(self.in_channels,),
            initializer=tf.constant_initializer(20.0),
            trainable=True,
            dtype=self.dtype or tf.float32
        )
        self.eps = 1e-10

    def call(self, x):
        # Compute L2 norm along channel axis (axis=3 for NHWC)
        norm = tf.sqrt(tf.reduce_sum(tf.pow(x, 2), axis=3, keepdims=True)) + self.eps
        x_norm = tf.math.truediv(x, norm)
        # Scale normalized output by learnable gamma (w)
        out = self.w * x_norm
        return out


class MyModel(tf.keras.Model):
    """
    SSD300 model wrapper with feature extractor, feature pyramid, heads,
    performing classification and bounding box regression.
    """

    def __init__(self):
        super(MyModel, self).__init__()
        self.num_classes = 21  # 20 classes + background

        # Build the feature extractor layers
        self.feature_extractor = create_feature_extractor(feature_extractor_config['300'])
        self.l2_norm = L2Norm(512, scale=20)
        self.feature_pyramid = create_feature_pyramid(feature_pyramid_config['300'], size=300)
        self.head = create_head(head_config['300'], self.num_classes)

    def call(self, x):
        """
        Forward pass:

        Runs input through VGG-like base network,
        applies L2 normalization to conv4_3 feature,
        appends features from pyramid layers,
        and produces regression and classification outputs.
        """
        # Assume input shape batch x 300 x 300 x 3
        features = []

        # Pass through feature_extractor layers 0-33 (first 34 layers)
        # This indexing comes from the original code's loop "for i in range(34):"
        for i in range(34):
            x = self.feature_extractor[i](x)
        # Apply L2 normalization to conv4_3 output (shape: batch x h x w x 512)
        s = self.l2_norm(x)
        features.append(s)

        # Pass remaining feature_extractor layers (34 to end)
        for i in range(34, len(self.feature_extractor)):
            x = self.feature_extractor[i](x)
        features.append(x)

        # Pass through feature pyramid layers, append specific outputs
        for k, v in enumerate(self.feature_pyramid):
            x = tf.nn.relu(v(x))
            # Append outputs at specific indices as in original code: 2,5,7,9
            if k in [2, 5, 7, 9]:
                features.append(x)

        regressions = []
        classifications = []

        # For each feature map, apply regression and classification heads
        for k, v in enumerate(features):
            # The original code uses batch size 32 explicitly in reshape. We make it generic.
            batch_size = tf.shape(v)[0]
            # Regression output shape: [batch_size, num_boxes, 4]
            reg = self.head['reg'][k](v)
            reg = tf.reshape(reg, [batch_size, -1, 4])
            regressions.append(reg)

            # Classification output shape: [batch_size, num_boxes, num_classes]
            cls = self.head['cls'][k](v)
            cls = tf.reshape(cls, [batch_size, -1, self.num_classes])
            classifications.append(cls)

        regressions = tf.concat(regressions, axis=1)
        classifications = tf.concat(classifications, axis=1)
        return [regressions, classifications]


def my_model_function():
    """
    Instantiate and return the SSD300 model.
    """
    return MyModel()


def GetInput():
    """
    Returns a random input tensor matching the expected model input shape:
    Batch size 32, 300 height, 300 width, 3 channels.
    Pixel values are floats in [0, 255].
    """
    return tf.random.uniform((32, 300, 300, 3), minval=0, maxval=255, dtype=tf.float32)

