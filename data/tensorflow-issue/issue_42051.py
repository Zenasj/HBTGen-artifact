# tf.random.uniform((B, 64, 64, 3), dtype=tf.float32) ← inferred input shape from training snippet

import tensorflow as tf
from tensorflow.keras import layers, backend

# Custom SyncBatchNormalization import alias to match original code pattern
BatchNormalization = tf.keras.layers.experimental.SyncBatchNormalization

# The code below integrates a minimal ResNet50 variant similar to the referenced code,
# but adapted inside a tf.keras.Model subclass and simplified for clarity.
# SyncBatchNormalization layers are used instead of regular BatchNormalization,
# reflecting the issue's context.
#
# Note: For brevity, only core building blocks of ResNet50 are implemented here:
#       block1 (bottleneck block), stack1 (stack of such blocks), and full ResNet50 model.
#
# The original issue replaced keras bn layers with SyncBatchNormalization in ResNet.
# The below model faithfully uses SyncBatchNormalization to replicate that.
#
# Input shape assumed: (batch, height=64, width=64, channels=3) from the example training data snippet.

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

        # Initial layers
        self.conv1_pad = layers.ZeroPadding2D(padding=((3,3),(3,3)), name='conv1_pad')
        self.conv1_conv = layers.Conv2D(64, kernel_size=7, strides=2, use_bias=True, name='conv1_conv')
        self.conv1_bn = BatchNormalization(axis=self.bn_axis, epsilon=1.001e-5, name='conv1_bn')
        self.conv1_relu = layers.Activation('relu', name='conv1_relu')
        self.pool1_pad = layers.ZeroPadding2D(padding=((1,1),(1,1)), name='pool1_pad')
        self.pool1_pool = layers.MaxPooling2D(pool_size=3, strides=2, name='pool1_pool')

        # Stacks of residual blocks
        self.stack2_conv2 = self._make_stack(filters=64, blocks=3, stride1=1, name='conv2')
        self.stack3_conv3 = self._make_stack(filters=128, blocks=4, stride1=2, name='conv3')
        self.stack4_conv4 = self._make_stack(filters=256, blocks=6, stride1=2, name='conv4')
        self.stack5_conv5 = self._make_stack(filters=512, blocks=3, stride1=2, name='conv5')

        # Final layers
        self.avg_pool = layers.GlobalAveragePooling2D(name='avg_pool')
        self.predictions = layers.Dense(1000, activation='softmax', name='predictions')

    def _bottleneck_block(self, x, filters, stride=1, conv_shortcut=True, name=None):
        """A residual bottleneck block using SyncBatchNormalization."""
        if conv_shortcut:
            shortcut = layers.Conv2D(4 * filters, 1, strides=stride, name=name + '_0_conv')(x)
            shortcut = BatchNormalization(axis=self.bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
        else:
            shortcut = x

        x = layers.Conv2D(filters, 1, strides=stride, name=name + '_1_conv')(x)
        x = BatchNormalization(axis=self.bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
        x = layers.Activation('relu', name=name + '_1_relu')(x)

        x = layers.Conv2D(filters, 3, padding='SAME', name=name + '_2_conv')(x)
        x = BatchNormalization(axis=self.bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
        x = layers.Activation('relu', name=name + '_2_relu')(x)

        x = layers.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
        x = BatchNormalization(axis=self.bn_axis, epsilon=1.001e-5, name=name + '_3_bn')(x)

        x = layers.Add(name=name + '_add')([shortcut, x])
        x = layers.Activation('relu', name=name + '_out')(x)
        return x

    def _make_stack(self, filters, blocks, stride1=2, name=None):
        def stack_fn(x):
            x = self._bottleneck_block(x, filters, stride=stride1, conv_shortcut=True, name=name + '_block1')
            for i in range(2, blocks + 1):
                x = self._bottleneck_block(x, filters, conv_shortcut=False, name=f'{name}_block{i}')
            return x
        return stack_fn

    def call(self, inputs, training=False):
        x = self.conv1_pad(inputs)
        x = self.conv1_conv(x)
        x = self.conv1_bn(x, training=training)
        x = self.conv1_relu(x)
        x = self.pool1_pad(x)
        x = self.pool1_pool(x)

        x = self.stack2_conv2(x)
        x = self.stack3_conv3(x)
        x = self.stack4_conv4(x)
        x = self.stack5_conv5(x)

        x = self.avg_pool(x)
        x = self.predictions(x)
        return x


def my_model_function():
    # Returns an instance of MyModel
    return MyModel()


def GetInput():
    # Generate a random input tensor matching the expected input shape
    # Based on example training tensor, shape is (batch, 64, 64, 3)
    # Use batch size of 8 arbitrarily
    return tf.random.uniform((8, 64, 64, 3), dtype=tf.float32)

