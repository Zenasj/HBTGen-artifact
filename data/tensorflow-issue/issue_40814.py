# tf.random.uniform((BATCH_SIZE, WIDTH, HEIGHT, CHANNELS), dtype=tf.float32)
# From the discussion, input shape is (batch_size, 256, 512, 2)

import tensorflow as tf

class Conv2dLayer(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, **kwargs):
        super().__init__(**kwargs)
        self.conv = tf.keras.layers.Conv2D(
            filters,
            kernel_size,
            strides=strides,
            padding="same",
            kernel_initializer="he_normal",
            dtype=tf.float32,  # Use float32 internally; mixed precision set externally
        )
        self.activation = tf.keras.layers.LeakyReLU()
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

    def call(self, inputs, training=None):
        x = self.conv(inputs)
        x = self.activation(x)
        x = self.batch_norm(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
        })
        return config

    def compute_output_shape(self, input_shape):
        return self.conv.compute_output_shape(input_shape)


class UpSampleLayer(tf.keras.layers.Layer):
    def __init__(self, filters, strides=2, **kwargs):
        super().__init__(**kwargs)
        self.upconv = tf.keras.layers.Conv2DTranspose(
            filters,
            4,
            strides=strides,
            padding="same",
            kernel_initializer="he_normal",
            dtype=tf.float32,
        )
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.activation = tf.keras.layers.LeakyReLU()
        self.filters = filters
        self.strides = strides

    def call(self, inputs, training=None):
        x = self.upconv(inputs)
        x = self.batch_norm(x, training=training)
        x = self.dropout(x, training=training)
        return self.activation(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "strides": self.strides,
        })
        return config


class DownsampleBlock(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = Conv2dLayer(filters, 4)
        self.conv2 = Conv2dLayer(filters, 4)
        self.downsample_conv = Conv2dLayer(filters, 4, strides=2)
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.filters = filters

    def call(self, inputs, training=None):
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)
        x = self.downsample_conv(x, training=training)
        x = self.dropout(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        return config


class MyModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mask_activation = tf.keras.layers.Activation("relu")
        self.axis = -1
        self.downsample_blocks = []
        self.upsample_blocks = []
        # Create downsample blocks from 16 to 512 filters doubling each time (6 blocks)
        # Store filters for upsample blocks in reverse order (except the last)
        n_maps_list = []
        for i in range(6):
            n_maps = 16 * (2 ** i)
            n_maps_list.insert(0, n_maps)  # prepend to reverse order
        
        for n_maps in n_maps_list[1:]:
            self.upsample_blocks.append(UpSampleLayer(n_maps, strides=2))

        # Append an extra UpsampleLayer with 2 filters and stride 2 at the end
        self.upsample_blocks.append(UpSampleLayer(2, strides=2))

        # Create Downsample blocks
        for n_maps in n_maps_list:
            self.downsample_blocks.append(DownsampleBlock(n_maps))

    def call(self, inputs, training=None):
        skip_connections = []
        x = inputs
        for down_block in self.downsample_blocks:
            x = down_block(x, training=training)
            skip_connections.insert(0, x)  # Prepend for reverse order

        # First upsample block (no skip connection)
        x = self.upsample_blocks[0](x, training=training)
        # Upsample blocks with skip connections concatenated along channels
        for up_block, skip in zip(self.upsample_blocks[1:], skip_connections[1:]):
            x = up_block(tf.keras.layers.concatenate([x, skip], axis=self.axis), training=training)

        return self.mask_activation(x)


def my_model_function():
    # Set mixed precision policy outside model to mimic issue environment
    policy = tf.keras.mixed_precision.Policy("mixed_float16")
    tf.keras.mixed_precision.set_global_policy(policy)

    model = MyModel()
    # Build model with fixed input shape for safety
    model.build(input_shape=(None, 256, 512, 2))
    # Compile with optimizer and loss for training
    model.compile(optimizer="adam", loss="mean_absolute_error")
    return model


def GetInput():
    # Create a random input tensor matching batch size 16, 256x512 spatial,
    # 2 channels, float32 dtype - which will be auto cast under mixed precision policy
    import numpy as np
    BATCH_SIZE = 16
    WIDTH = 256
    HEIGHT = 512
    CHANNELS = 2
    input_np = np.random.rand(BATCH_SIZE, WIDTH, HEIGHT, CHANNELS).astype("float32")
    return tf.convert_to_tensor(input_np)

