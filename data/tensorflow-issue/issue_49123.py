# tf.random.uniform((1, 224, 224, 3), dtype=tf.float32) ← Default input shape for EfficientNetB0

import tensorflow as tf
from tensorflow.keras import layers, Model

# Since the provided code is a partial EfficientNet implementation (mostly a factory and block building),
# we'll implement a simplified MyModel that uses EfficientNetB0 with dilation_rate=1,
# and another internal model with dilation_rate=2 for comparison,
# and return the ratio of output tensor sizes to reflect the performance impact discussed.

# We'll replicate the key behaviors:
# - EfficientNetB0-like block structure (simplified),
# - Replace core Conv2D in stem and blocks with explicit dilation_rate parameter,
# - Calculate output of both versions and compare output shapes to illustrate performance difference.

# Because the original code is large and partial, essential parts are inferred and simplified.

# Note: To properly compare Conv2D with/without dilation (and stride),
# we define two submodels in MyModel and compare their output sizes.

class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, dilation_rate=1, name=None):
        super().__init__(name=name)
        self.padding = layers.ZeroPadding2D(padding='same') if strides == 1 else layers.ZeroPadding2D(padding=1)
        self.conv = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                                  dilation_rate=dilation_rate, padding='valid' if strides > 1 else 'same',
                                  use_bias=False,
                                  kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_out', distribution='truncated_normal'))
        self.bn = layers.BatchNormalization()
        self.act = layers.Activation('swish')

    def call(self, x, training=False):
        x = self.padding(x)
        x = self.conv(x)
        x = self.bn(x, training=training)
        return self.act(x)

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Stem conv: filters=32, kernel=3, stride=2
        self.stem_no_dilation = ConvBlock(32, 3, strides=2, dilation_rate=1, name='stem_no_dilation')
        self.stem_dilation = ConvBlock(32, 3, strides=2, dilation_rate=2, name='stem_dilation')

        # One block to replicate block(…, kernel_size=3, strides=1) with/without dilation
        # For simplification, a single ConvBlock here
        self.block_no_dilation = ConvBlock(16, 3, strides=1, dilation_rate=1, name='block_no_dilation')
        self.block_dilation = ConvBlock(16, 3, strides=1, dilation_rate=2, name='block_dilation')

        # Global average pooling and final dense layer for classification
        self.global_pool = layers.GlobalAveragePooling2D()
        self.classifier = layers.Dense(1000, activation='softmax',
                                       kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1./3., mode='fan_out', distribution='uniform'))

    def call(self, inputs, training=False):
        # Run both models' stem + block
        x_no_dil = self.stem_no_dilation(inputs, training=training)
        x_no_dil = self.block_no_dilation(x_no_dil, training=training)
        out_no_dil = self.global_pool(x_no_dil)
        out_no_dil = self.classifier(out_no_dil)

        x_dil = self.stem_dilation(inputs, training=training)
        x_dil = self.block_dilation(x_dil, training=training)
        out_dil = self.global_pool(x_dil)
        out_dil = self.classifier(out_dil)

        # Compare outputs: here just output logits difference norm as proxy of differences
        diff = tf.reduce_mean(tf.abs(out_no_dil - out_dil))

        # Return the two outputs and difference metric
        return {
            "output_no_dilation": out_no_dil,
            "output_dilation": out_dil,
            "mean_abs_diff": diff,
            "shape_no_dilation": tf.shape(x_no_dil),
            "shape_dilation": tf.shape(x_dil),
        }

def my_model_function():
    return MyModel()

def GetInput():
    # EfficientNet typically expects input shape (batch, height, width, channels) = (1, 224, 224, 3)
    return tf.random.uniform((1, 224, 224, 3), dtype=tf.float32)

