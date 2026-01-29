# tf.random.uniform((BATCH_SIZE, 256, 256, 3), dtype=tf.float32)  # Inferred input shape for MCN (ImageNet style input)

import math
import tensorflow as tf
from tensorflow import keras

# Custom Swish activation layer with learnable weight and get_config for serialization
class Swish(keras.layers.Layer):
    def __init__(self):
        super(Swish, self).__init__()
        self.weight = self.add_weight(
            shape=(), initializer='uniform', trainable=True, name='weight')

    def call(self, inputs):
        return inputs + tf.sigmoid(self.weight * inputs)

    def get_config(self):
        base_config = super(Swish, self).get_config()
        # include the initial weight value if needed, currently we just rely on default serialization of weights
        return base_config


# Conv block: Conv2D + BatchNorm + Swish activation
class Conv(tf.keras.Model):
    def __init__(self, filters, kernel_size=1, strides=1, padding='valid'):
        super(Conv, self).__init__()
        self.conv = keras.layers.Conv2D(filters, kernel_size, strides, padding)
        self.bn = keras.layers.BatchNormalization()
        self.ac = Swish()

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.ac(x)
        return x

    def get_config(self):
        # Basic config for serialization; actual parameters could be extended if needed
        return {
            "filters": self.conv.filters,
            "kernel_size": self.conv.kernel_size,
            "strides": self.conv.strides,
            "padding": self.conv.padding,
        }


# Squeeze-and-Excitation (SE) block with get_config
class SEBlock(tf.keras.Model):
    def __init__(self, filters):
        super(SEBlock, self).__init__()
        self.conv0 = keras.layers.Conv2D(filters // 4, 1, 1)
        self.drop = keras.layers.Dropout(0.25)
        self.conv1 = keras.layers.Conv2D(filters, 1, 1)
        self.bn = keras.layers.BatchNormalization()
        self.ac = Swish()

    def call(self, inputs):
        # Global average pooling with reduce_mean over H, W dims
        pooled = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
        x = self.conv0(pooled)
        x = self.drop(x)
        x = self.conv1(x)
        x = self.bn(x)
        x = tf.sigmoid(x)
        x = x * inputs
        return self.ac(x)

    def get_config(self):
        return {"filters": self.conv1.filters}


# Residual block with Conv(1x1) + Dropout + Conv(3x3) + BatchNorm + Swish + residual connection
class ResBlock(tf.keras.Model):
    def __init__(self, filters):
        super(ResBlock, self).__init__()
        self.conv0 = keras.layers.Conv2D(filters // 4, 1, 1)
        self.drop = keras.layers.Dropout(0.25)
        self.conv1 = keras.layers.Conv2D(filters, 3, 1, 'same')
        self.bn = keras.layers.BatchNormalization()
        self.ac = Swish()

    def call(self, inputs):
        x = self.conv0(inputs)
        x = self.drop(x)
        x = self.conv1(x)
        x = self.bn(x)
        x = inputs + x
        x = self.ac(x)
        return x

    def get_config(self):
        return {
            "filters": self.conv1.filters,
        }


# The main fused model as MyModel, includes MCN520 architecture from the issue.
# This model is a Keras Model subclass that defines the MCN520 architecture.
class MyModel(tf.keras.Model):
    def __init__(self, width=2, growth=24, input_shape=(256, 256, 3)):
        super(MyModel, self).__init__()
        self.width = width
        self.growth = growth
        self.input_shape_ = input_shape

        # Initial channel scaling
        fs = int(width * growth)
        self.initial_conv = keras.layers.Conv2D(fs, 8, 2)
        self.initial_pool = keras.layers.MaxPool2D(2)
        
        # First SE and Res blocks layers after initial conv/pool
        self.seblock1 = SEBlock(fs)
        self.resblock1 = ResBlock(fs)
        self.conv1 = Conv(fs // width)

        self.conv2 = Conv(fs // width)

        # For progressive blocks based on the for loop [2,3,5,4], a list of layers
        self.blocks = []
        self.pools = []

        # We will build blocks dynamically in __init__ to match the "for i,depth in enumerate..." loop
        current_fs = fs
        current_width = width
        for i, depth in enumerate([2, 3, 5, 4]):
            block_layers = []
            for _ in range(int(6 * depth)):
                new_filters = int(math.sqrt(current_fs * current_width))
                current_fs += new_filters
                # Layers within each iteration
                block_layers.append(Conv(current_fs // width, 1, 1))
                block_layers.append(SEBlock(current_fs // width))
                block_layers.append(ResBlock(current_fs // width))
                block_layers.append(Conv(growth, 1, 1))  # growth filters

            self.blocks.append(block_layers)
            if i != 3:
                # pooling layers applied between blocks except last
                self.pools.append({
                    "conv": Conv(current_fs // 2),
                    "pool": keras.layers.MaxPool2D(2),
                    "se_conv": Conv((current_fs // 2) // width),
                    "se_pool": keras.layers.MaxPool2D(2),
                    "res_conv": Conv((current_fs // 2) // width),
                    "res_pool": keras.layers.MaxPool2D(2),
                })
                current_fs //= 2

        self.global_pool = keras.layers.GlobalMaxPool2D()
        self.dropout = keras.layers.Dropout(0.25)
        self.classifier = keras.layers.Dense(1000, activation='softmax')

    def call(self, inputs, training=None):
        # Initial conv and pool
        fs = int(self.width * self.growth)
        x = self.initial_conv(inputs)
        x = self.initial_pool(x)

        x1 = self.conv1(self.seblock1(x))
        x2 = self.conv2(self.resblock1(x))

        current_fs = fs
        current_width = self.width

        for i, block_layers in enumerate(self.blocks):
            for idx in range(0, len(block_layers), 4):
                # The original code concatenates and processes through layers:
                # concatenation of x, x1, x2
                t = tf.keras.layers.Concatenate()([x, x1, x2])
                t = tf.keras.layers.Dropout(0.25)(t, training=training)
                conv_layer = block_layers[idx]
                t = conv_layer(t)  # Conv with filters current_fs//width
                t = tf.keras.layers.Dropout(0.25)(t, training=training)
                se_layer = block_layers[idx + 1]
                res_layer = block_layers[idx + 2]
                t = se_layer(t)
                x1 = se_layer(t)
                x2 = res_layer(t)
                t = tf.keras.layers.Concatenate()([t, x1, x2])
                t = tf.keras.layers.Dropout(0.25)(t, training=training)
                conv_growth = block_layers[idx + 3]
                t = conv_growth(t)
                x = tf.keras.layers.Concatenate()([x, t])

                # Increase current_fs by sqrt(fs*width) every iteration equivalent is handled in loop init

            # After each depth block except last, do downsampling with conv and pooling on x, x1, x2
            if i != 3:
                pool_layer = self.pools[i]
                current_fs //= 2

                # For x
                x = pool_layer["pool"](pool_layer["conv"](x), training=training)

                # For x1
                x1 = pool_layer["se_pool"](pool_layer["se_conv"](x1), training=training)

                # For x2
                x2 = pool_layer["res_pool"](pool_layer["res_conv"](x2), training=training)

        x = self.global_pool(x)
        x = self.dropout(x, training=training)
        outputs = self.classifier(x)
        return outputs

    def get_config(self):
        return {
            "width": self.width,
            "growth": self.growth,
            "input_shape": self.input_shape_,
        }


def my_model_function():
    # Return an instance of MyModel with default parameters (width=2, growth=24, input shape 256x256x3)
    return MyModel(width=2, growth=24, input_shape=(256, 256, 3))


def GetInput():
    # Return a random float32 tensor of shape (batch_size, 256, 256, 3)
    # We assume batch size is 20 as per example
    batch_size = 20
    return tf.random.uniform((batch_size, 256, 256, 3), dtype=tf.float32)

