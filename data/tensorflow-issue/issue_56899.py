import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.lite.python import convert as _tf_convert
from tensorflow.lite.toco import types_pb2 as _types_pb2

def representative_data_gen():
        for i in range(10):
            yield [tf.random.normal((1, 640, 480, 3), dtype=tf.float32)]

def main():
    _input = x = tf.keras.layers.Input((640, 480, 3), dtype=tf.float32)
    x = tf.keras.layers.Conv2D(64, kernel_size=3)(x)
    output = tfa.layers.GELU(False, name='gelu')(x)
    tf_model = tf.keras.Model(inputs=[_input], outputs=[output])

    converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [ tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8 ]
    converter.representative_dataset = representative_data_gen
    int16_model = converter.convert()

    converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter._experimental_calibrate_only = True
    converter.representative_dataset = representative_data_gen
    calibrate_model = converter.convert()
    
    # _types_pb2.QUANTIZED_INT16
    quantized_model = _tf_convert.mlir_quantize(calibrate_model, inference_type=_types_pb2.QUANTIZED_INT16, denylisted_ops=['Gelu'])
    
    expect_quantized_model = _tf_convert.mlir_quantize(calibrate_model, denylisted_ops=['Gelu'])

    with open("int16_model.tflite", 'wb') as F:
        F.write(int16_model)

    with open("calibrate_model.tflite", 'wb') as F:
        F.write(calibrate_model)
    
    with open("quantized_model.tflite", 'wb') as F:
        F.write(quantized_model)
    
    with open("expect_quantized_model.tflite", 'wb') as F:
        F.write(expect_quantized_model)

if __name__ == "__main__":
    main()