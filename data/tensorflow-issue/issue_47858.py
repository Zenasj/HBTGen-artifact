from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import tensorflow as tf
from tensorflow import keras
import tensorflow_model_optimization as tfmot

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# Possible quantization aware quantizers:
QAT_ALL_VALUES = tfmot.quantization.keras.quantizers.AllValuesQuantizer
QAT_LAST_VALUE = tfmot.quantization.keras.quantizers.LastValueQuantizer
QAT_MA = tfmot.quantization.keras.quantizers.MovingAverageQuantizer


def quantization_aware_training(model, save, w_bits, a_bits, symmetric, per_axis, narrow_range, quantizer, batch_size=64, epochs=2):

    # Create quantized model's name string
    name = model.name + '_'
    name = name + str(w_bits) + 'wbits_' + str(a_bits) + 'abits_'

    if symmetric:
        name = name + 'sym_'
    else:
        name = name + 'asym_'

    if narrow_range:
        name = name + 'narr_'
    else:
        name = name + 'full_'

    if per_axis:
        name = name + 'perch_'
    else:
        name = name + 'perten_'

    if quantizer == QAT_ALL_VALUES:
        name = name + 'AV'
    elif quantizer == QAT_LAST_VALUE:
        name = name + 'LV'
    elif quantizer == QAT_MA:
        name = name + 'MA'

    # Quantization
    # *****
    quantize_apply = tfmot.quantization.keras.quantize_apply
    quantize_model = tfmot.quantization.keras.quantize_model
    quantize_annotate_layer = tfmot.quantization.keras.quantize_annotate_layer
    clone_model = tf.keras.models.clone_model
    quantize_scope = tfmot.quantization.keras.quantize_scope

    supported_layers = [
        tf.keras.layers.Conv2D,
    ]

    class Quantizer(tfmot.quantization.keras.QuantizeConfig):
        # Configure how to quantize weights.
        def get_weights_and_quantizers(self, layer):
            return [(layer.kernel, tfmot.quantization.keras.quantizers.LastValueQuantizer(num_bits=8, symmetric=True, narrow_range=False, per_axis=False))]

        # Configure how to quantize activations.
        def get_activations_and_quantizers(self, layer):
            return [(layer.activation, tfmot.quantization.keras.quantizers.MovingAverageQuantizer(num_bits=8, symmetric=False, narrow_range=False, per_axis=False))]

        def set_quantize_weights(self, layer, quantize_weights):
            # Add this line for each item returned in `get_weights_and_quantizers`
            # , in the same order
            layer.kernel = quantize_weights[0]

        def set_quantize_activations(self, layer, quantize_activations):
            # Add this line for each item returned in `get_activations_and_quantizers`
            # , in the same order.
            layer.activation = quantize_activations[0]

        # Configure how to quantize outputs (may be equivalent to activations).
        def get_output_quantizers(self, layer):
            return []

        def get_config(self):
            return {}

    class ConvQuantizer(Quantizer):
        # Configure weights to quantize with 4-bit instead of 8-bits.
        def get_weights_and_quantizers(self, layer):
            return [(layer.kernel, quantizer(num_bits=w_bits, symmetric=symmetric, narrow_range=narrow_range, per_axis=per_axis))]

        # Configure how to quantize activations.
        def get_activations_and_quantizers(self, layer):
            return [(layer.activation, tfmot.quantization.keras.quantizers.MovingAverageQuantizer(num_bits=a_bits, symmetric=False, narrow_range=False, per_axis=False))]

    class DepthwiseQuantizer(Quantizer):
        # Configure weights to quantize with 4-bit instead of 8-bits.
        def get_weights_and_quantizers(self, layer):
            return [(layer.depthwise_kernel, quantizer(num_bits=w_bits, symmetric=symmetric, narrow_range=narrow_range, per_axis=per_axis))]

        # Configure how to quantize activations.
        def get_activations_and_quantizers(self, layer):
            return [(layer.activation, tfmot.quantization.keras.quantizers.MovingAverageQuantizer(num_bits=a_bits, symmetric=False, narrow_range=False, per_axis=False))]

    # Instead of simply using quantize_annotate_model or quantize_model we must use
    # quantize_annotate_layer since it's the only one with a quantize_config argument
    def quantize_all_layers(layer):
        if isinstance(layer, tf.keras.layers.DepthwiseConv2D):
            return quantize_annotate_layer(layer, quantize_config=DepthwiseQuantizer())
        elif isinstance(layer, tf.keras.layers.Conv2D):
            return quantize_annotate_layer(layer, quantize_config=ConvQuantizer())
        return layer

    annotated_model = clone_model(
        model,
        clone_function=quantize_all_layers
    )

    with quantize_scope(
        {'Quantizer': Quantizer},
        {'ConvQuantizer': ConvQuantizer},
            {'DepthwiseQuantizer': DepthwiseQuantizer}):
        q_aware_model = quantize_apply(annotated_model)

    # *****

    # Compile and train model
    optimizer = keras.optimizers.Adam(
        learning_rate=0.001)
    q_aware_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True),
        optimizer=optimizer, metrics=['sparse_categorical_accuracy'])

    (train_images, train_labels),_ = keras.datasets.cifar10.load_data()

    q_aware_model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs, verbose=1,
                      validation_split=0.1)

    if save:
        save_path = 'models/temp/' + name
        q_aware_model.save(save_path + '.h5')

    return q_aware_model


def temp_net():
    dropout = 0.1

    model = keras.Sequential()
    model.add(keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(10, activation='softmax'))

    model._name = "temp_net"

    return model


if __name__ == "__main__":
    q_model = quantization_aware_training(model=temp_net(), save=True,
                                          w_bits=8, a_bits=8, symmetric=False, narrow_range=False, per_axis=False, quantizer=QAT_ALL_VALUES, batch_size=64, epochs=1)

import numpy as np
from tensorflow import keras
import tensorflow as tf
import tensorflow_model_optimization as tfmot


# Possible quantization aware quantizers:
QAT_ALL_VALUES = tfmot.quantization.keras.quantizers.AllValuesQuantizer
QAT_LAST_VALUE = tfmot.quantization.keras.quantizers.LastValueQuantizer
QAT_MA = tfmot.quantization.keras.quantizers.MovingAverageQuantizer


def _get_QAT_model(model, w_bits, a_bits, symmetric, per_axis, narrow_range, quantizer):
    quantize_apply = tfmot.quantization.keras.quantize_apply
    quantize_model = tfmot.quantization.keras.quantize_model
    quantize_annotate_layer = tfmot.quantization.keras.quantize_annotate_layer
    clone_model = tf.keras.models.clone_model
    quantize_scope = tfmot.quantization.keras.quantize_scope

    class Quantizer(tfmot.quantization.keras.QuantizeConfig):
        # Configure how to quantize weights.
        def get_weights_and_quantizers(self, layer):
            return [(layer.kernel, tfmot.quantization.keras.quantizers.LastValueQuantizer(num_bits=8, symmetric=True, narrow_range=False, per_axis=False))]

        # Configure how to quantize activations.
        def get_activations_and_quantizers(self, layer):
            return [(layer.activation, tfmot.quantization.keras.quantizers.MovingAverageQuantizer(num_bits=8, symmetric=False, narrow_range=False, per_axis=False))]

        def set_quantize_weights(self, layer, quantize_weights):
            # Add this line for each item returned in `get_weights_and_quantizers`
            # , in the same order
            layer.kernel = quantize_weights[0]

        def set_quantize_activations(self, layer, quantize_activations):
            # Add this line for each item returned in `get_activations_and_quantizers`
            # , in the same order.
            layer.activation = quantize_activations[0]

        # Configure how to quantize outputs (may be equivalent to activations).
        def get_output_quantizers(self, layer):
            return []

        def get_config(self):
            return {}

    class ConvQuantizer(Quantizer):
        # Configure how to quantize weights
        def get_weights_and_quantizers(self, layer):
            return [(layer.kernel, quantizer(num_bits=w_bits, symmetric=symmetric, narrow_range=narrow_range, per_axis=per_axis))]

        # Configure how to quantize activations.
        def get_activations_and_quantizers(self, layer):
            return [(layer.activation, tfmot.quantization.keras.quantizers.MovingAverageQuantizer(num_bits=a_bits, symmetric=False, narrow_range=False, per_axis=False))]

    class DenseQuantizer(Quantizer):
        # Configure how to quantize weights
        def get_weights_and_quantizers(self, layer):
            return [(layer.kernel, quantizer(num_bits=w_bits, symmetric=symmetric, narrow_range=narrow_range, per_axis=per_axis))]

        # Configure how to quantize activations.
        def get_activations_and_quantizers(self, layer):
            return [(layer.activation, tfmot.quantization.keras.quantizers.MovingAverageQuantizer(num_bits=a_bits, symmetric=False, narrow_range=False, per_axis=False))]

    def quantize_all_layers(layer):
        if isinstance(layer, keras.layers.Conv2D):
            return quantize_annotate_layer(layer, quantize_config=ConvQuantizer())
        if isinstance(layer, keras.layers.Dense):
            return quantize_annotate_layer(layer, quantize_config=DenseQuantizer())
        return layer

    annotated_model = clone_model(
        model,
        clone_function=quantize_all_layers
    )

    with quantize_scope(
        {'Quantizer': Quantizer},
        {'ConvQuantizer': ConvQuantizer},
            {'DenseQuantizer': DenseQuantizer}):
        q_aware_model = quantize_apply(annotated_model)

    return q_aware_model


def test_net():
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(
        32, (3, 3), padding='same', input_shape=(32, 32, 3)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.Conv2D(10, (1, 1), padding='same'))
    model.add(keras.layers.Activation('softmax'))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(10, activation='softmax'))

    model._name = "test_net"

    return model


if __name__ == "__main__":
    model = test_net()
    # When setting per_axis=False it works. When per_axis=True an error occurs.
    model = _get_QAT_model(model, w_bits=8, a_bits=8, symmetric=True,
                           per_axis=True, narrow_range=False, quantizer=QAT_LAST_VALUE)
    print(model.summary())

class ConvQuantizer(Quantizer):
        # Configure how to quantize weights
        def get_weights_and_quantizers(self, layer):
            return [(layer.kernel, ConvWeightsQuantizer(num_bits=w_bits, symmetric=symmetric, narrow_range=narrow_range, per_axis=per_axis))]

        # Configure how to quantize activations.
        def get_activations_and_quantizers(self, layer):
            return [(layer.activation, tfmot.quantization.keras.quantizers.MovingAverageQuantizer(num_bits=a_bits, symmetric=False, narrow_range=False, per_axis=False))]

import numpy as np
from tensorflow import keras
import tensorflow as tf
import tensorflow_model_optimization as tfmot


# Possible quantization aware quantizers:
QAT_ALL_VALUES = tfmot.quantization.keras.quantizers.AllValuesQuantizer
QAT_LAST_VALUE = tfmot.quantization.keras.quantizers.LastValueQuantizer
QAT_MA = tfmot.quantization.keras.quantizers.MovingAverageQuantizer


def _get_QAT_model(model, w_bits, a_bits, symmetric, per_axis, narrow_range, quantizer):
    quantize_apply = tfmot.quantization.keras.quantize_apply
    quantize_model = tfmot.quantization.keras.quantize_model
    quantize_annotate_layer = tfmot.quantization.keras.quantize_annotate_layer
    clone_model = tf.keras.models.clone_model
    quantize_scope = tfmot.quantization.keras.quantize_scope

    class Quantizer(tfmot.quantization.keras.QuantizeConfig):
        # Configure how to quantize weights.
        def get_weights_and_quantizers(self, layer):
            return [(layer.kernel, tfmot.quantization.keras.quantizers.LastValueQuantizer(num_bits=8, symmetric=True, narrow_range=False, per_axis=False))]

        # Configure how to quantize activations.
        def get_activations_and_quantizers(self, layer):
            return [(layer.activation, tfmot.quantization.keras.quantizers.MovingAverageQuantizer(num_bits=8, symmetric=False, narrow_range=False, per_axis=False))]

        def set_quantize_weights(self, layer, quantize_weights):
            # Add this line for each item returned in `get_weights_and_quantizers`
            # , in the same order
            layer.kernel = quantize_weights[0]

        def set_quantize_activations(self, layer, quantize_activations):
            # Add this line for each item returned in `get_activations_and_quantizers`
            # , in the same order.
            layer.activation = quantize_activations[0]

        # Configure how to quantize outputs (may be equivalent to activations).
        def get_output_quantizers(self, layer):
            return []

        def get_config(self):
            return {}

    class ConvWeightsQuantizer(QAT_LAST_VALUE):

        def build(self, tensor_shape, name, layer):
            min_weight = layer.add_weight(
                name + '_min',
                shape=(tensor_shape[-1],),
                initializer=tf.keras.initializers.Constant(-6.0),
                trainable=False)
            max_weight = layer.add_weight(
                name + '_max',
                shape=(tensor_shape[-1],),
                initializer=tf.keras.initializers.Constant(6.0),
                trainable=False)
            return {'min_var': min_weight, 'max_var': max_weight}

    class ConvQuantizer(Quantizer):
        # Configure how to quantize weights
        def get_weights_and_quantizers(self, layer):
            return [(layer.kernel, ConvWeightsQuantizer(num_bits=w_bits, symmetric=symmetric, narrow_range=narrow_range, per_axis=per_axis))]

        # Configure how to quantize activations.
        def get_activations_and_quantizers(self, layer):
            return [(layer.activation, tfmot.quantization.keras.quantizers.MovingAverageQuantizer(num_bits=a_bits, symmetric=False, narrow_range=False, per_axis=False))]

    class DepthwiseQuantizer(Quantizer):
        # Configure how to quantize weights
        def get_weights_and_quantizers(self, layer):
            return [(layer.depthwise_kernel, ConvWeightsQuantizer(num_bits=w_bits, symmetric=symmetric, narrow_range=narrow_range, per_axis=per_axis))]

        # Configure how to quantize activations.
        def get_activations_and_quantizers(self, layer):
            return [(layer.activation, tfmot.quantization.keras.quantizers.MovingAverageQuantizer(num_bits=a_bits, symmetric=False, narrow_range=False, per_axis=False))]

    class DenseQuantizer(Quantizer):
        # Configure how to quantize weights
        def get_weights_and_quantizers(self, layer):
            return [(layer.kernel, ConvWeightsQuantizer(num_bits=w_bits, symmetric=symmetric, narrow_range=narrow_range, per_axis=per_axis))]

        # Configure how to quantize activations.
        def get_activations_and_quantizers(self, layer):
            return [(layer.activation, tfmot.quantization.keras.quantizers.MovingAverageQuantizer(num_bits=a_bits, symmetric=False, narrow_range=False, per_axis=False))]

    def quantize_all_layers(layer):
        if isinstance(layer, keras.layers.DepthwiseConv2D):
            return quantize_annotate_layer(layer, quantize_config=DepthwiseQuantizer())
        elif isinstance(layer, keras.layers.Conv2D):
            return quantize_annotate_layer(layer, quantize_config=ConvQuantizer())
        elif isinstance(layer, keras.layers.Dense):
            return quantize_annotate_layer(layer, quantize_config=DenseQuantizer())
        return layer

    annotated_model = clone_model(
        model,
        clone_function=quantize_all_layers
    )

    with quantize_scope(
        {'Quantizer': Quantizer},
        {'ConvQuantizer': ConvQuantizer},
        {'DepthwiseQuantizer': DepthwiseQuantizer},
            {'DenseQuantizer': DenseQuantizer}):
        q_aware_model = quantize_apply(annotated_model)

    return q_aware_model


def test_net():
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(
        32, (3, 3), padding='same', input_shape=(32, 32, 3)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))

    model.add(keras.layers.DepthwiseConv2D(1, 1))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(10, activation='softmax'))

    model._name = "test_net"

    return model


def print_kernel_info(model):
    for i, layer in enumerate(model.layers):
        if 'depthwise_conv' in layer.name:
            variables = layer.variables
            print("kernel_min: " +
                  str(np.float(variables[4])) + ", kernel_max: " + str(np.float(variables[5])))


if __name__ == "__main__":
    model = test_net()
    # When setting per_axis=False it works. When per_axis=True an error occurs.
    model = _get_QAT_model(model, w_bits=8, a_bits=8, symmetric=True,
                           per_axis=True, narrow_range=False, quantizer=QAT_LAST_VALUE)

    print_kernel_info(model)