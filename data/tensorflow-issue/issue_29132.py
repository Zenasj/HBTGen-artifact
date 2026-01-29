# tf.random.uniform((B, 224, 224, 3), dtype=tf.float32)
import tensorflow as tf

class ResidualBlock(tf.keras.Model):
    def __init__(self, block_type=None, n_filters=None):
        super(ResidualBlock, self).__init__()
        self.n_filters = n_filters
        if block_type == 'identity':
            self.strides = 1
        elif block_type == 'conv':
            self.strides = 2
            self.conv_shortcut = tf.keras.layers.Conv2D(
                filters=self.n_filters,
                kernel_size=1,
                padding='same',
                strides=self.strides,
                kernel_initializer='he_normal')
            self.bn_shortcut = tf.keras.layers.BatchNormalization(momentum=0.9)
        else:
            # Default to identity if unspecified
            self.strides = 1

        self.conv_1 = tf.keras.layers.Conv2D(
            filters=self.n_filters,
            kernel_size=3,
            padding='same',
            strides=self.strides,
            kernel_initializer='he_normal')
        self.bn_1 = tf.keras.layers.BatchNormalization(momentum=0.9)
        self.relu_1 = tf.keras.layers.ReLU()

        self.conv_2 = tf.keras.layers.Conv2D(
            filters=self.n_filters,
            kernel_size=3,
            padding='same',
            kernel_initializer='he_normal')
        self.bn_2 = tf.keras.layers.BatchNormalization(momentum=0.9)
        self.relu_2 = tf.keras.layers.ReLU()

    def call(self, x, training=False):
        shortcut = x
        if self.strides == 2:
            shortcut = self.conv_shortcut(x)
            shortcut = self.bn_shortcut(shortcut, training=training)

        y = self.conv_1(x)
        y = self.bn_1(y, training=training)
        y = self.relu_1(y)
        y = self.conv_2(y)
        y = self.bn_2(y, training=training)
        y = tf.add(shortcut, y)
        y = self.relu_2(y)
        return y


class ResNet34(tf.keras.Model):
    def __init__(self, include_top=True, n_classes=1000):
        super(ResNet34, self).__init__()

        self.n_classes = n_classes
        self.include_top = include_top

        self.conv_1 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=7,
            padding='same',
            strides=2,
            kernel_initializer='he_normal')
        self.bn_1 = tf.keras.layers.BatchNormalization(momentum=0.9)
        self.relu_1 = tf.keras.layers.ReLU()
        self.maxpool = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')

        self.residual_blocks = tf.keras.Sequential()
        # ResNet34 configuration: filters, number of blocks, downscale first block
        for n_filters, reps, downscale in zip(
            [64, 128, 256, 512],
            [3, 4, 6, 3],
            [False, True, True, True]):
            for i in range(reps):
                if i == 0 and downscale:
                    self.residual_blocks.add(ResidualBlock(block_type='conv', n_filters=n_filters))
                else:
                    self.residual_blocks.add(ResidualBlock(block_type='identity', n_filters=n_filters))

        self.GAP = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(units=self.n_classes)

    def call(self, x, training=False):
        y = self.conv_1(x)
        y = self.bn_1(y, training=training)
        y = self.relu_1(y)
        y = self.maxpool(y)
        y = self.residual_blocks(y, training=training)
        if self.include_top:
            y = self.GAP(y)
            y = self.fc(y)
        return y


class MyModel(tf.keras.Model):
    """
    This MyModel class fuses the ResNet34 subclassed model from the issue into a single callable model.
    It encapsulates the layers and provides a clean interface with proper build and shape handling.
    """

    def __init__(self, include_top=True, n_classes=1000):
        super(MyModel, self).__init__()
        self.resnet34 = ResNet34(include_top=include_top, n_classes=n_classes)

    def call(self, inputs, training=False):
        return self.resnet34(inputs, training=training)

    def build_graph(self, input_shape):
        """
        Helper function to build the model and create a tf.keras.Input for shape inference and summary.
        This mimics the workaround to get meaningful model.summary() output for subclassed models.
        """
        input_shape_nobatch = input_shape[1:]
        self.build(input_shape)
        inputs = tf.keras.Input(shape=input_shape_nobatch)
        _ = self.call(inputs)
        return tf.keras.Model(inputs=inputs, outputs=_)


def my_model_function():
    """
    Returns an instance of MyModel configured like ResNet34 with top dense layer for 1000 classes.
    """
    model = MyModel(include_top=True, n_classes=1000)
    # Build the model model to initialize weights and get correct shapes.
    model.build((None, 224, 224, 3))
    return model


def GetInput():
    """
    Returns a random tensor input matching the expected input of MyModel.
    Shape: (Batch, Height, Width, Channels) = (1, 224, 224, 3)
    dtype: tf.float32
    """
    return tf.random.uniform((1, 224, 224, 3), dtype=tf.float32)

