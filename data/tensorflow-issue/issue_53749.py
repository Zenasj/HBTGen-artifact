from tensorflow.keras import models

import tensorflow as tf
from tensorflow import keras
import tensorflow_model_optimization as tfmot
import numpy as np
import tempfile
from tensorflow.keras import layers


LastValueQuantizer = tfmot.quantization.keras.quantizers.LastValueQuantizer
MovingAverageQuantizer = tfmot.quantization.keras.quantizers.MovingAverageQuantizer

class DefaultConv2DQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    def get_weights_and_quantizers(self, layer):
        return [(layer.kernel, LastValueQuantizer(num_bits=8, symmetric=True, narrow_range=False, per_axis=False))]
    def get_activations_and_quantizers(self, layer):
        return [(layer.activation, MovingAverageQuantizer(num_bits=16, symmetric=False, narrow_range=False, per_axis=False))]
    def set_quantize_weights(self, layer, quantize_weights):
        layer.kernel = quantize_weights[0]
    def set_quantize_activations(self, layer, quantize_activations):
        layer.activation = quantize_activations[0]
    def get_output_quantizers(self, layer):
        return []
    def get_config(self):
        return {}

def apply_quantization_to_conv2d(layer):
    if isinstance(layer, tf.keras.layers.Conv2D):
        return   tfmot.quantization.keras.quantize_annotate_layer(layer, DefaultConv2DQuantizeConfig())
    return layer


if __name__ == "__main__":
    quantize_annotate_layer = tfmot.quantization.keras.quantize_annotate_layer
    quantize_annotate_model = tfmot.quantization.keras.quantize_annotate_model
    quantize_scope = tfmot.quantization.keras.quantize_scope

    input = keras.Input(shape=(28,28,1))
    x1 = keras.layers.Conv2D(filters=12, kernel_size=(3,3), activation='relu')(input)
    x2 = keras.layers.Conv2D(filters=12, kernel_size=(3,3), activation='relu')(x1)
    x3 = keras.layers.Conv2D(filters=12, kernel_size=(3,3), activation='relu')(x2)

    model = keras.Model(input, x3)

    model = tf.keras.models.clone_model(model, clone_function=apply_quantization_to_conv2d)
    with tfmot.quantization.keras.quantize_scope({'DefaultConv2DQuantizeConfig': DefaultConv2DQuantizeConfig}):
        model = tfmot.quantization.keras.quantize_apply(model)

    q_aware_model = model
    converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    quantized_tflite_model = converter.convert()

    #save tflite file
    with open("testtf25.tflite", 'wb') as f:
        f.write(quantized_tflite_model)

    interpreter = tf.lite.Interpreter(model_content=quantized_tflite_model)
    interpreter.allocate_tensors()