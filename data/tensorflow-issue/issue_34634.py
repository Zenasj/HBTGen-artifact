# tf.random.uniform((B, 256, 256, 3), dtype=tf.float32)

import math
import tensorflow as tf
from tensorflow import keras

class Swish(keras.layers.Layer):
    def __init__(self):
        super(Swish, self).__init__()
        # Trainable scalar weight initialized uniformly
        self.weight = self.add_weight(initializer='uniform', trainable=True)

    def __call__(self, inputs):
        # Swish activation: inputs + sigmoid(weight * inputs)
        return inputs + tf.sigmoid(self.weight * inputs)

class Conv(keras.Model):
    def __init__(self, filters, kernel_size=1, strides=1, padding='valid'):
        super(Conv, self).__init__()
        self.conv = keras.layers.Conv2D(filters, kernel_size, strides, padding)
        self.bn = keras.layers.BatchNormalization()
        self.ac = Swish()

    def __call__(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        return self.ac(x)

class SEBlock(keras.Model):
    def __init__(self, filters):
        super(SEBlock, self).__init__()
        # Squeeze and Excitation block with Conv2D bottleneck and scaling
        self.conv0 = keras.layers.Conv2D(filters // 4, 1, 1)
        self.drop = keras.layers.Dropout(0.25)
        self.conv1 = keras.layers.Conv2D(filters, 1, 1)
        self.bn = keras.layers.BatchNormalization()
        self.ac = Swish()

    def __call__(self, inputs):
        # Global spatial average pooling + bottleneck + sigmoid gating + scaling
        se = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
        se = self.conv0(se)
        se = self.drop(se)
        se = self.conv1(se)
        se = tf.sigmoid(se)
        scaled = se * inputs
        bn_out = self.bn(scaled)
        return self.ac(bn_out)

class ResBlock(keras.Model):
    def __init__(self, filters):
        super(ResBlock, self).__init__()
        # Residual block with bottleneck convs and Swish activation
        self.conv0 = keras.layers.Conv2D(filters // 4, 1, 1)
        self.drop = keras.layers.Dropout(0.25)
        self.conv1 = keras.layers.Conv2D(filters, 3, 1, padding='same')
        self.bn = keras.layers.BatchNormalization()
        self.ac = Swish()

    def __call__(self, inputs):
        x = self.conv0(inputs)
        x = self.drop(x)
        x = self.conv1(x)
        x = self.bn(x)
        x = inputs + x  # Residual connection
        return self.ac(x)

class MyModel(tf.keras.Model):
    def __init__(self, width=2, growth=24, input_shape=(256, 256, 3)):
        super(MyModel, self).__init__()
        # Initialized similar to the functional mcn_520, but implemented as subclassed model
        
        self.width = width
        self.growth = growth
        self.input_spec = tf.keras.layers.InputSpec(shape=(None, *input_shape))
        self.input_shape_ = input_shape
        
        fs = int(width * growth)
        self._fs = fs  # Keep as state for building blocks
        
        # Initial layers
        self.conv_init = keras.layers.Conv2D(fs, 8, 2)
        self.pool_init = keras.layers.MaxPool2D(2)
        
        # SEBlock, ResBlock, and Conv wrappers on first stage
        self.se_block = SEBlock(fs)
        self.res_block = ResBlock(fs)
        self.conv1 = Conv(fs // width)
        self.conv2 = Conv(fs // width)

        # For iterative layers, need dynamic creation
        # We'll store layers lists to build dynamically in call
        # Because filters "fs" increases over iterations, we'll create Conv, SEBlock, ResBlock dynamically
        
        self.dropout = keras.layers.Dropout(0.25)
        self.concat = keras.layers.Concatenate()
        self.global_pool = keras.layers.GlobalMaxPool2D()
        self.fc = keras.layers.Dense(1000, activation='softmax')

    @tf.function(jit_compile=True)
    def call(self, inputs, training=False):
        width = self.width
        growth = self.growth
        fs = self._fs

        x = self.conv_init(inputs)
        x = self.pool_init(x)
        x1 = self.conv1(self.se_block(x))
        x2 = self.conv2(self.res_block(x))
        
        for i, depth in enumerate([2, 3, 5, 4]):
            for _ in range(int(6 * depth)):
                fs += int(math.sqrt(fs * width))
                t = self.concat([x, x1, x2])
                t = self.dropout(t, training=training)
                conv = Conv(fs // width, 1, 1)
                t = conv(t)
                t = self.dropout(t, training=training)
                se_block = SEBlock(fs // width)
                res_block = ResBlock(fs // width)
                x1 = se_block(t)
                x2 = res_block(t)
                t = self.concat([t, x1, x2])
                t = self.dropout(t, training=training)
                conv_growth = Conv(growth, 1, 1)
                t = conv_growth(t)
                x = self.concat([x, t])
            if i != 3:
                fs //= 2
                conv_fs = Conv(fs)
                conv_fs_x = conv_fs(x)
                x = keras.layers.MaxPool2D(2)(conv_fs_x)
                conv_fs_x1 = Conv(fs // width)
                conv_fs_x1_out = conv_fs_x1(x1)
                x1 = keras.layers.MaxPool2D(2)(conv_fs_x1_out)
                conv_fs_x2 = Conv(fs // width)
                conv_fs_x2_out = conv_fs_x2(x2)
                x2 = keras.layers.MaxPool2D(2)(conv_fs_x2_out)
        
        x = self.global_pool(x)
        x = self.dropout(x, training=training)
        out = self.fc(x)
        return out

def my_model_function():
    # Return an instance of MyModel with default parameters matching the original functional model
    return MyModel(width=2, growth=24, input_shape=(256, 256, 3))

def GetInput():
    # Provide a random input tensor with batch size 1, height=256, width=256, channels=3 (dtype float32)
    return tf.random.uniform((1, 256, 256, 3), dtype=tf.float32)

