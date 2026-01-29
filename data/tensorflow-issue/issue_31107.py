# tf.random.uniform((1, 800, 1600, 3), dtype=tf.float32) ← inferred from buildFastScnn input shape

import tensorflow as tf

def conv_block(inputs, conv_type, filter_count, kernel_size, strides, padding='same', relu=True):
    if conv_type == 'ds':
        x = tf.keras.layers.SeparableConv2D(filter_count, kernel_size, padding=padding, strides=strides)(inputs)
    else:
        x = tf.keras.layers.Conv2D(filter_count, kernel_size, padding=padding, strides=strides)(inputs)

    x = tf.keras.layers.BatchNormalization()(x)

    if relu:
        x = tf.keras.activations.relu(x)

    return x


def _res_bottleneck(inputs, filters, kernel, t, s, r=False):
    tchannel = tf.keras.backend.int_shape(inputs)[-1] * t

    x = conv_block(inputs, 'conv', tchannel, (1, 1), strides=(1, 1))

    x = tf.keras.layers.DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.activations.relu(x)

    x = conv_block(x, 'conv', filters, (1, 1), strides=(1, 1), padding='same', relu=False)

    if r:
        x = tf.keras.layers.add([x, inputs])
    return x


def bottleneck_block(inputs, filters, kernel, t, strides, n):
    x = _res_bottleneck(inputs, filters, kernel, t, strides)

    for i in range(1, n):
        x = _res_bottleneck(x, filters, kernel, t, 1, True)

    return x


def pyramid_pooling_block(input_tensor, bin_sizes, input_height, input_width):
    # The spatial downsampled size of input tensor is assumed 1/32 original dims
    width = input_width // 32
    height = input_height // 32

    concat_list = [input_tensor]
    for bin_size in bin_sizes:
        # Average pool with stride and pool size determined by bin size relative to width/height
        x = tf.keras.layers.AveragePooling2D(pool_size=(height // bin_size, width // bin_size),
                                             strides=(height // bin_size, width // bin_size))(input_tensor)
        x = tf.keras.layers.Conv2D(128, 3, 2, padding='same')(x)
        # Resize back to the downscaled height and width
        x = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, (height, width)))(x)

        concat_list.append(x)

    return tf.keras.layers.concatenate(concat_list)


class MyModel(tf.keras.Model):
    def __init__(self, input_height=800, input_width=1600, input_channel=3, n_classes=20, weights_path=None):
        super().__init__()

        self.input_height = input_height
        self.input_width = input_width
        self.input_channel = input_channel
        self.n_classes = n_classes

        # Initial conv_blocks chain (learning deeper spatial features)
        self.conv1 = lambda inp: conv_block(inp, 'conv', 32, (3,3), strides=(2,2))
        self.conv2 = lambda inp: conv_block(inp, 'ds', 48, (3,3), strides=(2,2))
        self.conv3 = lambda inp: conv_block(inp, 'ds', 64, (3,3), strides=(2,2))

        # Bottleneck blocks as keras layers for GFE (Global Feature Extractor)
        # Note: each bottleneck block uses _res_bottleneck internally
        self.bottleneck1 = None
        self.bottleneck2 = None
        self.bottleneck3 = None

        # Build the bottleneck blocks as separate models since function logic cannot be directly used in __init__
        # We'll build them in build() method because they require input shapes.

        # Pyramid pooling will be built in call as it depends on tensor shape

        # Layers for feature fusion in call:
        self.ff_conv = None  # Conv block 1x1 with 128 filters, relu=False, to be created in build
        self.ff_upsample = tf.keras.layers.UpSampling2D((4,4))
        self.ff_depthwise = tf.keras.layers.DepthwiseConv2D((3,3), strides=(1,1), depth_multiplier=1, padding='same')
        self.ff_bn = tf.keras.layers.BatchNormalization()
        self.ff_relu = tf.keras.layers.Activation('relu')
        self.ff_conv2 = tf.keras.layers.Conv2D(128, 1, 1, padding='same', activation=None)  # no activation here
        self.ff_bn2 = tf.keras.layers.BatchNormalization()
        self.ff_relu2 = tf.keras.layers.Activation('relu')

        # Classifier layers:
        self.classifier_dsconv1 = tf.keras.layers.SeparableConv2D(128, (3,3), padding='same', strides=(1,1))
        self.classifier_bn1 = tf.keras.layers.BatchNormalization()
        self.classifier_act1 = tf.keras.layers.Activation('relu')

        self.classifier_dsconv2 = tf.keras.layers.SeparableConv2D(128, (3,3), padding='same', strides=(1,1))
        self.classifier_bn2 = tf.keras.layers.BatchNormalization()
        self.classifier_act2 = tf.keras.layers.Activation('relu')

        self.classifier_conv = tf.keras.layers.Conv2D(n_classes, (1,1), padding='same', strides=(1,1))
        self.classifier_bn3 = tf.keras.layers.BatchNormalization()
        self.classifier_act3 = tf.keras.layers.Activation('relu')

        self.classifier_dropout = tf.keras.layers.Dropout(0.3)
        self.classifier_upsample = tf.keras.layers.UpSampling2D((8,8))
        self.classifier_softmax = tf.keras.layers.Softmax()

        # Save weights path for optional loading after build
        self.weights_path = weights_path

        # Mark built flag for bottleneck blocks
        self._bottlenecks_built = False

    def build(self, input_shape):
        # Build bottleneck blocks now that we know input shapes:
        dummy_input = tf.keras.Input(shape=input_shape[1:])
        x = self.conv1(dummy_input)
        x = self.conv2(x)
        x = self.conv3(x)

        self.bottleneck1 = self._make_bottleneck_block(x, 64, (3,3), 6, 2, 3)
        x = self.bottleneck1.output
        self.bottleneck2 = self._make_bottleneck_block(x, 96, (3,3), 6, 2, 3)
        x = self.bottleneck2.output
        self.bottleneck3 = self._make_bottleneck_block(x, 128, (3,3), 6, 1, 3)

        # Weights loading handled after build by caller or after initialization

        # Create ff_conv 1x1 with 128 filters relu=False on feature fusion layer
        self.ff_conv = tf.keras.layers.Conv2D(128, (1,1), padding='same', strides=(1,1), activation=None)

        self._bottlenecks_built = True

        super().build(input_shape)

    def _make_bottleneck_block(self, inputs, filters, kernel, t, strides, n):
        # Construct bottleneck block Keras Model for reuse
        x = _res_bottleneck(inputs, filters, kernel, t, strides)
        for i in range(1, n):
            x = _res_bottleneck(x, filters, kernel, t, 1, True)
        model = tf.keras.Model(inputs=inputs, outputs=x)
        return model

    def call(self, inputs, training=False):
        # Encoder path
        lds_layer = self.conv1(inputs)
        lds_layer = self.conv2(lds_layer)
        lds_layer = self.conv3(lds_layer)

        # Global Feature Extractor - bottleneck blocks
        if not self._bottlenecks_built:
            # Lazily build on first call if not built
            self.build(tf.shape(inputs))

        gfe_layer = self.bottleneck1(lds_layer)
        gfe_layer = self.bottleneck2(gfe_layer)
        gfe_layer = self.bottleneck3(gfe_layer)

        # Pyramid pooling block - custom composed here
        gfe_layer = pyramid_pooling_block(gfe_layer,
                                          bin_sizes=[2,4,6,8],
                                          input_height=self.input_height,
                                          input_width=self.input_width)

        # Feature Fusion Layer
        ff_layer1 = conv_block(lds_layer, 'conv', 128, (1,1), padding='same', strides=(1,1), relu=False)
        ff_layer2 = self.ff_upsample(gfe_layer)
        ff_layer2 = self.ff_depthwise(ff_layer2)
        ff_layer2 = self.ff_bn(ff_layer2, training=training)
        ff_layer2 = tf.keras.activations.relu(ff_layer2)
        ff_layer2 = self.ff_conv2(ff_layer2)

        ff_final = tf.keras.layers.add([ff_layer1, ff_layer2])
        ff_final = tf.keras.layers.BatchNormalization()(ff_final)
        ff_final = tf.keras.activations.relu(ff_final)

        # Classifier
        classifier = self.classifier_dsconv1(ff_final)
        classifier = self.classifier_bn1(classifier, training=training)
        classifier = self.classifier_act1(classifier)

        classifier = self.classifier_dsconv2(classifier)
        classifier = self.classifier_bn2(classifier, training=training)
        classifier = self.classifier_act2(classifier)

        classifier = self.classifier_conv(classifier)
        classifier = self.classifier_bn3(classifier, training=training)
        classifier = self.classifier_act3(classifier)

        classifier = self.classifier_dropout(classifier, training=training)
        classifier = self.classifier_upsample(classifier)
        classifier = self.classifier_softmax(classifier)

        return classifier

    def get_config(self):
        config = super().get_config()
        config.update({
            'input_height': self.input_height,
            'input_width': self.input_width,
            'input_channel': self.input_channel,
            'n_classes': self.n_classes,
            'weights_path': self.weights_path,
        })
        return config


def my_model_function():
    # Instantiate and return the MyModel instance with typical default args matching original code
    model = MyModel(input_height=800, input_width=1600, input_channel=3, n_classes=20, weights_path=None)
    return model


def GetInput():
    # Return a random input tensor matching the model’s expected input shape (batch=1 for simplicity)
    return tf.random.uniform((1, 800, 1600, 3), dtype=tf.float32)

