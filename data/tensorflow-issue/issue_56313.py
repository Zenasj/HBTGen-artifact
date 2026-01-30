import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import optimizers

import tensorflow_model_optimization as tfmot

LastValueQuantizer = tfmot.quantization.keras.quantizers.LastValueQuantizer
MovingAverageQuantizer = tfmot.quantization.keras.quantizers.MovingAverageQuantizer

class DefaultDenseQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    # List all of your weights
    weights = {
        "kernel": LastValueQuantizer(num_bits=8, symmetric=True, narrow_range=False, per_axis=False)
    }

    # List of all your activations
    activations = {
        "activation": MovingAverageQuantizer(num_bits=8, symmetric=False, narrow_range=False, per_axis=False)
    }

    # Configure how to quantize weights.
    def get_weights_and_quantizers(self, layer):
        output = []
        for attribute, quantizer in self.weights.items():
            if hasattr(layer, attribute):
                output.append((getattr(layer, attribute), quantizer))

        return output

    # Configure how to quantize activations.
    def get_activations_and_quantizers(self, layer):
        output = []
        for attribute, quantizer in self.activations.items():
            if hasattr(layer, attribute):
                output.append((getattr(layer, attribute), quantizer))

        return output

    def set_quantize_weights(self, layer, quantize_weights):
        # Add this line for each item returned in `get_weights_and_quantizers`
        # , in the same order

        count = 0
        for attribute in self.weights.keys():
            if hasattr(layer, attribute):
                setattr(layer, attribute, quantize_weights[count])
                count += 1

    def set_quantize_activations(self, layer, quantize_activations):
        # Add this line for each item returned in `get_activations_and_quantizers`
        # , in the same order.
        count = 0
        for attribute in self.activations.keys():
            if hasattr(layer, attribute):
                setattr(layer, attribute, quantize_activations[count])
                count += 1

    # Configure how to quantize outputs (may be equivalent to activations).
    def get_output_quantizers(self, layer):
        return []

    def get_config(self):
        return {}


from quant import DefaultDenseQuantizeConfig
from tensorflow_model_optimization.python.core.quantization.keras.quantize import quantize_scope, quantize_apply
import tensorflow_model_optimization as tfmot


with quantize_scope({
    "DefaultDenseQuantizeConfig": DefaultDenseQuantizeConfig,
    "CustomLayer": CustomLayer
}):
    def apply_quantization_to_layer(layer):
        return tfmot.quantization.keras.quantize_annotate_layer(layer, DefaultDenseQuantizeConfig())

    annotated_model = tf.keras.models.clone_model(
        tflite_model,
        clone_function=apply_quantization_to_layer,
    )

    qat_model = tfmot.quantization.keras.quantize_apply(annotated_model)

    qat_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=['accuracy']
    )

    qat_model.summary()

def representative_dataset():
    for _ in range(1000):
        data = np.random.choice([-32768, 0, 1, 32767], size=(88200,))  #give it a  fixed size , Dynamic size is still an obstacle in some cases during TFLite conversion
        yield [data.astype(np.uint8)] # uint8/int8 recommended while trying uint8 quantization


converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
converter.experimental_new_converter = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS_INT8, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.# resolves select ops issues
]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_quant_model = converter.convert()

def representative_dataset():
    for _ in range(1000):
        data = np.random.choice([-32768, 0, 1, 32767], size=(88200,))  #give it a  fixed size , Dynamic size is still an obstacle in some cases during TFLite conversion
        yield [data.astype(np.float32)] # uint8/int8 recommended while trying uint8 quantization


converter = tf.lite.TFLiteConverter.from_keras_model(qat_model)
converter.experimental_new_converter = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.# resolves select ops issues
]
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32
tflite_quant_model = converter.convert()