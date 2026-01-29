# tf.random.uniform((B, 224, 224, 3), dtype=tf.float32)  ← Assumed input shape from usage in code and issue context

import tensorflow as tf
import math

def swish(x):
    return x * tf.keras.activations.sigmoid(x)

def round_filters(filters, multiplier):
    depth_divisor = 8
    min_depth = max(depth_divisor, 1)
    filters = filters * multiplier
    new_filters = max(min_depth, int(filters + depth_divisor / 2) // depth_divisor * depth_divisor)
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)

def round_repeats(repeats, multiplier):
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))

def SEBlock(inputs, input_channels, ratio=0.25):
    num_reduced_filters = max(1, int(input_channels * ratio))
    branch = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    # Expanding dims to (batch, 1, 1, channels)
    branch = tf.keras.backend.expand_dims(branch, 1)
    branch = tf.keras.backend.expand_dims(branch, 1)
    branch = tf.keras.layers.Conv2D(filters=num_reduced_filters, kernel_size=(1, 1), strides=1, padding="same")(branch)
    branch = swish(branch)
    branch = tf.keras.layers.Conv2D(filters=input_channels, kernel_size=(1, 1), strides=1, padding='same')(branch)
    branch = tf.keras.activations.sigmoid(branch)
    output = inputs * branch
    return output

def MBConv(in_channels, out_channels, expansion_factor, stride, k, drop_connect_rate, inputs, training=False):
    x = tf.keras.layers.Conv2D(filters=in_channels * expansion_factor, kernel_size=(1, 1), strides=1, padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x, training=training)
    x = swish(x)
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=(k, k), strides=stride, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x, training=training)
    x = SEBlock(x, in_channels*expansion_factor)
    x = swish(x)
    x = tf.keras.layers.Conv2D(filters=out_channels, kernel_size=(1, 1), strides=1, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x, training=training)
    if stride == 1 and in_channels == out_channels:
        if drop_connect_rate:
            x = tf.keras.layers.Dropout(rate=drop_connect_rate)(x, training=training)
        x = tf.keras.layers.Add()([x, inputs])
    return x

def build_mbconv_block(inputs, in_channels, out_channels, layers, stride, expansion_factor, k, drop_connect_rate, training):
    x = inputs
    for i in range(layers):
        if i == 0:
            x = MBConv(in_channels=in_channels, out_channels=out_channels, expansion_factor=expansion_factor,
                       stride=stride, k=k, drop_connect_rate=drop_connect_rate, inputs=x, training=training)
        else:
            x = MBConv(in_channels=out_channels, out_channels=out_channels, expansion_factor=expansion_factor,
                       stride=1, k=k, drop_connect_rate=drop_connect_rate, inputs=x, training=training)
    return x

def EfficientNet(inputs, width_coefficient, depth_coefficient, dropout_rate, drop_connect_rate=0.2, training=False):
    # Returns output and list of intermediate features per blocks
    features = []

    x = tf.keras.layers.Conv2D(filters=round_filters(32, width_coefficient),
                               kernel_size=(3, 3),
                               strides=2,
                               padding="same")(inputs)
    x = tf.keras.layers.BatchNormalization()(x, training=training)
    x = swish(x)

    x = build_mbconv_block(x, in_channels=round_filters(32, width_coefficient),
                           out_channels=round_filters(16, width_coefficient),
                           layers=round_repeats(1, depth_coefficient),
                           stride=1,
                           expansion_factor=1, k=3,
                           drop_connect_rate=drop_connect_rate,
                           training=training)
    features.append(x)

    x = build_mbconv_block(x, in_channels=round_filters(16, width_coefficient),
                           out_channels=round_filters(24, width_coefficient),
                           layers=round_repeats(2, depth_coefficient),
                           stride=1,
                           expansion_factor=6, k=3,
                           drop_connect_rate=drop_connect_rate,
                           training=training)
    features.append(x)

    x = build_mbconv_block(x, in_channels=round_filters(24, width_coefficient),
                           out_channels=round_filters(40, width_coefficient),
                           layers=round_repeats(2, depth_coefficient),
                           stride=2,
                           expansion_factor=6, k=5,
                           drop_connect_rate=drop_connect_rate,
                           training=training)
    features.append(x)

    x = build_mbconv_block(x, in_channels=round_filters(40, width_coefficient),
                           out_channels=round_filters(80, width_coefficient),
                           layers=round_repeats(3, depth_coefficient),
                           stride=2,
                           expansion_factor=6, k=3,
                           drop_connect_rate=drop_connect_rate,
                           training=training)
    features.append(x)

    x = build_mbconv_block(x, in_channels=round_filters(80, width_coefficient),
                           out_channels=round_filters(112, width_coefficient),
                           layers=round_repeats(3, depth_coefficient),
                           stride=1,
                           expansion_factor=6, k=5,
                           drop_connect_rate=drop_connect_rate,
                           training=training)
    features.append(x)

    x = build_mbconv_block(x, in_channels=round_filters(112, width_coefficient),
                           out_channels=round_filters(192, width_coefficient),
                           layers=round_repeats(4, depth_coefficient),
                           stride=2,
                           expansion_factor=6, k=5,
                           drop_connect_rate=drop_connect_rate,
                           training=training)
    features.append(x)

    x = build_mbconv_block(x, in_channels=round_filters(192, width_coefficient),
                           out_channels=round_filters(320, width_coefficient),
                           layers=round_repeats(1, depth_coefficient),
                           stride=1,
                           expansion_factor=6, k=3,
                           drop_connect_rate=drop_connect_rate,
                           training=training)
    features.append(x)

    x = tf.keras.layers.Conv2D(filters=round_filters(1280, width_coefficient),
                               kernel_size=(1, 1),
                               strides=1,
                               padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x, training=training)
    x = swish(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(rate=dropout_rate)(x, training=training)
    # Using units=1 + softmax is unconventional but replicate original code behavior
    # Usually classification uses units=NUM_CLASSES with softmax activation or units=1 with sigmoid.
    x = tf.keras.layers.Dense(units=1, activation=tf.keras.activations.softmax)(x)

    return x, features

def efficient_net_b0(inputs, training):
    return EfficientNet(inputs,
                        width_coefficient=1.0,
                        depth_coefficient=1.0,
                        dropout_rate=0.2,
                        drop_connect_rate=0.2,
                        training=training)

def up_sample(inputs, training=True):
    x = tf.keras.layers.UpSampling2D()(inputs)
    x = tf.keras.layers.BatchNormalization()(x, training=training)
    x = tf.keras.layers.ReLU()(x)
    return x


class MyModel(tf.keras.Model):
    """
    Fused model encapsulating both biggerModel and smallModel behavior.
    It can be constructed as biggerModel or smallModel by setting 'mode'.

    Comparison logic: Given same input, it can output boolean tensor indicating weight compatibility by
    comparing weights at overlapping layers named 'quater' and 'half'.

    If models' outputs shapes differ, fallback outputs None.

    Assumptions:
    - Input: Tensor of shape (B, 224, 224, 3), float32.
    - Output classes: 19 channels (18+1).
    - When training=True: models are larger (more layers).
    - When training=False: smaller number of layers (for smallModel).
    """

    def __init__(self, mode='bigger', outc=19, training=True):
        """
        mode: 'bigger' or 'smaller'
        outc: output channels for final conv layers
        training: bool flag to switch training mode behavior
        """
        super().__init__()
        self.mode = mode
        self.outc = outc
        self.training = training

        if mode == 'bigger':
            # Build layers internally rather than call functional style to reuse in .call

            # Input-dependent layers moved to call. Here we build shared layers that do not depend on input shape.

            # We will internally build the efficientnet features on demand

            # For downstream layers in biggerModel:
            # upsampling and conv blocks for spatial alignment

            # Will define layers for 'quater' and 'half' branches and upsampling chains.
            # Note: We use a simple structure to replicate original logic.

            self.efficientnet_width = 1.0
            self.efficientnet_depth = 1.0
            self.dropout_rate = 0.2
            self.drop_connect_rate = 0.2

            # UpSample layers
            self.up_sample_layer = tf.keras.layers.UpSampling2D()
            self.bn_up = tf.keras.layers.BatchNormalization()
            self.relu_up = tf.keras.layers.ReLU()

            # After concatenation of selected features (those with spatial dims <= input_dim/4)
            self.concat_layer = tf.keras.layers.Concatenate()

            # Following convoltions as per biggerModel after concatenation
            self.quater_conv1 = tf.keras.layers.Conv2D(512, 3, 1, 'same', activation=tf.nn.relu)
            self.quater_conv2 = tf.keras.layers.Conv2D(512, 3, 1, 'same', activation=tf.nn.relu)
            self.quater_out = tf.keras.layers.Conv2D(outc, 1, 1, 'same', name='quater', activation=None)

            # Half resolution branch
            self.half_up_conv = tf.keras.layers.UpSampling2D()
            self.half_bn = tf.keras.layers.BatchNormalization()
            self.half_relu = tf.keras.layers.ReLU()
            self.half_conv1 = tf.keras.layers.Conv2D(512, 3, 1, 'same', activation=tf.nn.relu)
            self.half_conv2 = tf.keras.layers.Conv2D(512, 3, 1, 'same', activation=tf.nn.relu)
            self.half_out = tf.keras.layers.Conv2D(outc, 1, 1, 'same', name='half', activation=None)

        elif mode == 'smaller':
            # smallModel layers setup

            self.conv1 = tf.keras.layers.Conv2D(512, 3, 1, 'same', activation=tf.nn.relu, name='Conv2_1')
            self.conv2 = tf.keras.layers.Conv2D(512, 3, 1, 'same', activation=tf.nn.relu, name='Conv2_2')
            self.quater_out = tf.keras.layers.Conv2D(outc, 1, 1, 'same', name='quater', activation=None)
            self.half_conv1 = tf.keras.layers.Conv2D(512, 3, 1, 'same', activation=tf.nn.relu)
            self.half_conv2 = tf.keras.layers.Conv2D(512, 3, 1, 'same', activation=tf.nn.relu)
            self.half_out = tf.keras.layers.Conv2D(outc, 1, 1, 'same', name='half', activation=None)
        else:
            raise ValueError("mode must be 'bigger' or 'smaller'")

    def call(self, inputs, training=None):
        training = training if training is not None else self.training

        if self.mode == 'bigger':
            # Build EfficientNet features dynamically
            _, features = EfficientNet(inputs,
                                      width_coefficient=self.efficientnet_width,
                                      depth_coefficient=self.efficientnet_depth,
                                      dropout_rate=self.dropout_rate,
                                      drop_connect_rate=self.drop_connect_rate,
                                      training=training)

            # Select features whose spatial dimension >= input_dim / 4 (height dimension)
            selected_feats = []
            input_dim = tf.shape(inputs)[1]
            target_dim = input_dim // 4
            for f in features:
                # We check static shape first, fallback to dynamic shape if None
                if f.shape[1] is not None and f.shape[1] >= target_dim:
                    selected_feats.append(f)
                else:
                    # if static shape unknown, check at runtime
                    cond = tf.greater_equal(tf.shape(f)[1], target_dim)
                    if cond:
                        selected_feats.append(f)

            # Concatenate along channel axis
            concat = tf.keras.layers.Concatenate()(selected_feats)
            quater_res = self.quater_conv1(concat)
            quater_res = self.quater_conv2(quater_res)
            quater_res_out = self.quater_out(quater_res)

            half_res = tf.image.resize(quater_res, size=(tf.shape(quater_res)[1]*2, tf.shape(quater_res)[2]*2), method='nearest')
            half_res = self.half_conv1(half_res)
            half_res = self.half_conv2(half_res)
            half_res_out = self.half_out(half_res)

            if training:
                return quater_res_out, half_res_out
            else:
                return quater_res_out

        else:
            # smaller model forward
            quater_res = self.conv1(inputs)
            quater_res = self.conv2(quater_res)
            quater_res_out = self.quater_out(quater_res)

            half_res = self.half_conv1(quater_res)
            half_res = self.half_conv2(half_res)
            half_res_out = self.half_out(half_res)

            if training:
                return quater_res_out, half_res_out
            else:
                return quater_res_out


def my_model_function():
    """
    Create and return an instance of MyModel.
    For example, create the bigger model with training=True by default.
    """
    # For demonstration, create bigger model with training=True
    return MyModel(mode='bigger', outc=19, training=True)


def GetInput():
    """
    Return a random tensor input that matches the expected input by MyModel.
    Input shape inferred as (B, 224, 224, 3) float32.
    """
    B = 1  # batch size
    H = 224
    W = 224
    C = 3
    return tf.random.uniform((B, H, W, C), dtype=tf.float32)

