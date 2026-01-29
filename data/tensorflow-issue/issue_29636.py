# tf.random.uniform((B, 512, 512, 3), dtype=tf.float32) ← Input shape inferred from SCALED_HEIGHT, SCALED_WIDTH, and num_channels=3

import tensorflow as tf

# For TensorFlow 2.0+ compatibility, group norm is not in tf.contrib anymore.
# The original issue is using tf.contrib.layers.group_norm which fails on DeferredTensor.
# We replace group_norm with a custom GroupNormalization layer implementation.

class GroupNormalization(tf.keras.layers.Layer):
    def __init__(self, groups=32, axis=-1, epsilon=1e-5, **kwargs):
        super(GroupNormalization, self).__init__(**kwargs)
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon

    def build(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError("Axis dimension of the inputs should be known. Found None.")
        if dim % self.groups != 0:
            raise ValueError(
                f"Number of groups ({self.groups}) must divide the number of channels ({dim})"
            )
        self.gamma = self.add_weight(
            shape=(dim,),
            initializer="ones",
            trainable=True,
            name="gamma"
        )
        self.beta = self.add_weight(
            shape=(dim,),
            initializer="zeros",
            trainable=True,
            name="beta"
        )
        super(GroupNormalization, self).build(input_shape)

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        tensor_shape = inputs.get_shape().as_list()
        # Prepare broadcast shape for gamma and beta
        ndim = len(tensor_shape)

        # reshape inputs to (N, groups, group_channels, ...)
        group_shape = [input_shape[0], self.groups, tensor_shape[self.axis] // self.groups]
        # Other dims flattened into spatial_dims
        spatial_dims = []
        for i in range(1, ndim):
            if i != self.axis:
                spatial_dims.append(input_shape[i])
        group_shape.extend(spatial_dims)

        # Reshape inputs
        group_axes = [1, 2] + list(range(3, ndim+1))  # axes over groups + channels + spatial
        inputs = tf.reshape(inputs, group_shape)

        # Calculate mean and variance along group channels and spatial dims
        mean, var = tf.nn.moments(inputs, axes=[2] + list(range(3, tf.rank(inputs))), keepdims=True)

        inputs = (inputs - mean) / tf.sqrt(var + self.epsilon)

        # Reshape back
        inputs = tf.reshape(inputs, input_shape)

        # Apply gamma and beta
        gamma = tf.reshape(self.gamma, [1 if i != self.axis else tensor_shape[self.axis] for i in range(ndim)])
        beta = tf.reshape(self.beta, [1 if i != self.axis else tensor_shape[self.axis] for i in range(ndim)])

        return inputs * gamma + beta

    def get_config(self):
        config = super(GroupNormalization, self).get_config()
        config.update({
            "groups": self.groups,
            "axis": self.axis,
            "epsilon": self.epsilon,
        })
        return config


def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters
    conv_name_base = 'res{}{}_branch'.format(stage, block)
    bn_name_base = 'bn{}{}_branch'.format(stage, block)

    x = tf.keras.layers.Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2c')(x)

    x = tf.keras.layers.add([x, input_tensor])
    x = tf.keras.layers.Activation('relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2,2)):

    filters1, filters2, filters3 = filters
    conv_name_base = 'res{}{}_branch'.format(stage, block)
    bn_name_base = 'bn{}{}_branch'.format(stage, block)

    x = tf.keras.layers.Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a')(input_tensor)
    x = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2c')(x)

    shortcut = tf.keras.layers.Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base + '1')(input_tensor)
    shortcut = tf.keras.layers.BatchNormalization(axis=3, name=bn_name_base + '1')(shortcut)

    x = tf.keras.layers.add([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)
    return x

def get_fpn(num_channels=3):

    input_ = tf.keras.Input(shape=(512, 512, num_channels))
    x = tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1')(input_)
    x = tf.keras.layers.BatchNormalization(axis=3, name='bn_conv1')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    bottomup_xl = identity_block(x, 3, [64, 64, 256], stage=2, block='c') # (None, 127, 127, 256)

    x = conv_block(bottomup_xl, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    bottomup_l = identity_block(x, 3, [128, 128, 512], stage=3, block='d') # (None, 64, 64, 512)

    x = conv_block(bottomup_l, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    bottomup_m = identity_block(x, 3, [256, 256, 1024], stage=4, block='f') # (None, 32, 32, 1024)

    x = conv_block(bottomup_m, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    bottomup_s = identity_block(x, 3, [512, 512, 2048], stage=5, block='c') # (None, 16, 16, 2048)

    topdown_s = tf.keras.layers.Conv2D(1024, (1, 1), padding='same')(bottomup_s)
    x = tf.keras.layers.UpSampling2D(size=(2, 2))(topdown_s)
    topdown_m = tf.keras.layers.add([x, tf.keras.layers.Conv2D(1024, (1, 1), padding='same')(bottomup_m)])

    x = tf.keras.layers.UpSampling2D(size=(2, 2))(topdown_m)
    topdown_l = tf.keras.layers.add([x, tf.keras.layers.Conv2D(1024, (1, 1), padding='same')(bottomup_l)])

    x = tf.keras.layers.UpSampling2D(size=(2, 2))(topdown_l)
    topdown_xl = tf.keras.layers.add([x, tf.keras.layers.Conv2D(1024, (1, 1), padding='same')(bottomup_xl)])

    pyramid_s = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(topdown_s)
    pyramid_m = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(topdown_m)
    pyramid_l = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(topdown_l)
    pyramid_xl = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(topdown_xl)

    return tf.keras.Model(input_, [pyramid_s, pyramid_m, pyramid_l, pyramid_xl])

NUM_LABELS = 21  # Assumed number of output channels for segmentation prediction; adjust as needed


class MyModel(tf.keras.Model):
    def __init__(self, num_channels=3, num_labels=NUM_LABELS, groups=32):
        super(MyModel, self).__init__()
        self.num_channels = num_channels
        self.num_labels = num_labels
        self.groups = groups
        self.fpn = get_fpn(num_channels)

        # Use GroupNormalization replacing tf.contrib.layers.group_norm
        # Create Conv2D for combined output after upsampling and summation
        self.conv_final = tf.keras.layers.Conv2D(num_labels, (1, 1), padding='same')

    def upsample(self, tensor, repetitions):
        """Upsample block applying Conv + GroupNorm + ReLU + (optional upsampling)."""
        x = tensor
        for i in range(repetitions):
            x = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(x)
            x = GroupNormalization(groups=self.groups, axis=-1)(x)
            x = tf.keras.layers.ReLU()(x)
            x = tf.image.resize(x, (tf.shape(x)[1] * 2, tf.shape(x)[2] * 2), method='nearest')
        if repetitions == 0:
            # If no upsampling, just conv + group norm + relu
            x = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(x)
            x = GroupNormalization(groups=self.groups, axis=-1)(x)
            x = tf.keras.layers.ReLU()(x)
        return x

    @tf.function(jit_compile=True)
    def call(self, inputs, training=False):
        pyramid_s, pyramid_m, pyramid_l, pyramid_xl = self.fpn(inputs)
        from_s = self.upsample(pyramid_s, 3)
        from_m = self.upsample(pyramid_m, 2)
        from_l = self.upsample(pyramid_l, 1)
        from_xl = self.upsample(pyramid_xl, 0)

        x = tf.keras.layers.add([from_s, from_m, from_l, from_xl])
        x = self.conv_final(x)
        # Upsample final output by factor 4 (nearest neighbor)
        shape = tf.shape(x)
        output_size = (shape[1] * 4, shape[2] * 4)
        output = tf.image.resize(x, output_size, method='nearest')
        return output


def my_model_function():
    # Return an instance of the MyModel with default parameters
    return MyModel()


def GetInput():
    # Return a random tensor input matching shape (batch_size, 512, 512, 3)
    # batch size chosen as 2 for example
    batch_size = 2
    input_tensor = tf.random.uniform(
        (batch_size, 512, 512, 3), dtype=tf.float32, minval=0, maxval=1)
    return input_tensor

