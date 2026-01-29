# tf.random.uniform((64, 32, 32, 3), dtype=tf.float32)  ← Assumed batch size 64, height and width 32, channels 3 for test input

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_model_optimization as tfmot

# This code is based on the issue discussion about Quantization Aware Training (QAT)
# and its support for per-axis (per-channel) quantization using the LastValueQuantizer.
# It provides a fused model supporting Conv2D, DepthwiseConv2D, and Dense layers quantized,
# using a custom weights quantizer that creates per-channel min/max variables when per_axis=True.
#
# The original problem: QuantizeWrapper kernel_min / kernel_max were scalar for per_axis=True,
# causing shape incompatibility errors during quantization.
#
# The solution inferred and implemented here is a ConvWeightsQuantizer class that overrides
# the build method to create min/max per channel variables of shape (tensor_shape[-1],),
# enabling proper per-channel quantization semantics with LastValueQuantizer.
#
# This model structure and quantize config can be used as a starting point for QAT with per-axis quantization
# support in Conv, DepthwiseConv, and Dense layers.
#
# The forward pass returns model logits, and input shape is fixed to (batch, 32, 32, 3) as in examples.

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Construct a small test network as specified in the issue discussion
        self.conv1 = keras.layers.Conv2D(32, (3,3), padding='same', input_shape=(32,32,3), name='conv2d_1')
        self.bn1 = keras.layers.BatchNormalization()
        self.act1 = keras.layers.Activation('relu')
        self.dwconv = keras.layers.DepthwiseConv2D(1, 1, name='depthwise_conv2d')
        self.flatten = keras.layers.Flatten()
        self.dense = keras.layers.Dense(10, activation='softmax', name='dense_output')

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.act1(x)
        x = self.dwconv(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x


def MyConvWeightsQuantizerClass(num_bits=8, symmetric=True, narrow_range=False, per_axis=False):
    """
    Factory function returning a subclass of LastValueQuantizer that overrides build()
    to create min/max per-channel variables when per_axis=True.
    This replicates the class ConvWeightsQuantizer from the issue discussion.
    """
    base = tfmot.quantization.keras.quantizers.LastValueQuantizer

    class ConvWeightsQuantizer(base):
        def __init__(self):
            super().__init__(
                num_bits=num_bits,
                symmetric=symmetric,
                narrow_range=narrow_range,
                per_axis=per_axis
            )

        def build(self, tensor_shape, name, layer):
            # Build min/max variables per channel if per_axis True, else scalar.
            if self.per_axis:
                # tensor_shape is e.g. (kernel_h, kernel_w, input_ch, output_ch) for Conv2D
                # Typically we quantize per output channel (last dim)
                min_weight = layer.add_weight(
                    name + '_min',
                    shape=(tensor_shape[-1],),
                    initializer=tf.keras.initializers.Constant(-6.0),
                    trainable=False
                )
                max_weight = layer.add_weight(
                    name + '_max',
                    shape=(tensor_shape[-1],),
                    initializer=tf.keras.initializers.Constant(6.0),
                    trainable=False
                )
            else:
                # scalar min/max variables
                min_weight = layer.add_weight(
                    name + '_min',
                    shape=(),
                    initializer=tf.keras.initializers.Constant(-6.0),
                    trainable=False
                )
                max_weight = layer.add_weight(
                    name + '_max',
                    shape=(),
                    initializer=tf.keras.initializers.Constant(6.0),
                    trainable=False
                )
            return {'min_var': min_weight, 'max_var': max_weight}

    return ConvWeightsQuantizer


def my_model_function():
    """
    Returns an instance of MyModel with quantization-aware training wrappers applied
    according to the discussed quantizers for per_axis quantization.
    """
    # Hyperparameters for quantization - these can be adjusted as needed.
    w_bits = 8
    a_bits = 8
    symmetric = True
    narrow_range = False
    per_axis = True  # This enables per-channel quantization

    QAT_LAST_VALUE = tfmot.quantization.keras.quantizers.LastValueQuantizer
    QAT_MA = tfmot.quantization.keras.quantizers.MovingAverageQuantizer

    # Custom quantize config base class to share common logic
    class QuantizeConfigBase(tfmot.quantization.keras.QuantizeConfig):
        def set_quantize_weights(self, layer, quantize_weights):
            # Expecting one weight to quantize
            layer.kernel = quantize_weights[0]

        def set_quantize_activations(self, layer, quantize_activations):
            # Expecting one activation to quantize
            layer.activation = quantize_activations[0]

        def get_output_quantizers(self, layer):
            return []

        def get_config(self):
            return {}

    # Instantiate ConvWeightsQuantizer for weights quantization
    ConvWeightsQuantizer = MyConvWeightsQuantizerClass(
        num_bits=w_bits,
        symmetric=symmetric,
        narrow_range=narrow_range,
        per_axis=per_axis
    )

    class ConvQuantizer(QuantizeConfigBase):
        def get_weights_and_quantizers(self, layer):
            return [(layer.kernel, ConvWeightsQuantizer())]

        def get_activations_and_quantizers(self, layer):
            # Use MovingAverageQuantizer with per_axis=False for activations as per discussion
            return [(layer.activation, QAT_MA(num_bits=a_bits, symmetric=False, narrow_range=False, per_axis=False))]

    class DepthwiseQuantizer(QuantizeConfigBase):
        def get_weights_and_quantizers(self, layer):
            return [(layer.depthwise_kernel, ConvWeightsQuantizer())]

        def get_activations_and_quantizers(self, layer):
            return [(layer.activation, QAT_MA(num_bits=a_bits, symmetric=False, narrow_range=False, per_axis=False))]

    class DenseQuantizer(QuantizeConfigBase):
        def get_weights_and_quantizers(self, layer):
            return [(layer.kernel, ConvWeightsQuantizer())]

        def get_activations_and_quantizers(self, layer):
            return [(layer.activation, QAT_MA(num_bits=a_bits, symmetric=False, narrow_range=False, per_axis=False))]

    # Function to annotate each layer type accordingly
    def quantize_all_layers(layer):
        if isinstance(layer, keras.layers.DepthwiseConv2D):
            return tfmot.quantization.keras.quantize_annotate_layer(layer, quantize_config=DepthwiseQuantizer())
        elif isinstance(layer, keras.layers.Conv2D):
            return tfmot.quantization.keras.quantize_annotate_layer(layer, quantize_config=ConvQuantizer())
        elif isinstance(layer, keras.layers.Dense):
            return tfmot.quantization.keras.quantize_annotate_layer(layer, quantize_config=DenseQuantizer())
        return layer

    base_model = MyModel()

    # Clone model with quantize annotations
    annotated_model = tf.keras.models.clone_model(
        base_model,
        clone_function=quantize_all_layers
    )

    # Scope dictionary required by quantize_scope
    with tfmot.quantization.keras.quantize_scope(
        {'QuantizeConfigBase': QuantizeConfigBase,
         'ConvQuantizer': ConvQuantizer,
         'DepthwiseQuantizer': DepthwiseQuantizer,
         'DenseQuantizer': DenseQuantizer,
         'ConvWeightsQuantizer': ConvWeightsQuantizer}):
        # Apply quantization transforms
        qat_model = tfmot.quantization.keras.quantize_apply(annotated_model)

    return qat_model


def GetInput():
    """
    Returns a random input tensor matching the expected input shape of MyModel,
    with batch size 64, 32 height, 32 width, and 3 channels (RGB image).
    """
    return tf.random.uniform((64, 32, 32, 3), dtype=tf.float32)

