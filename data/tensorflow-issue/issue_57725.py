import math
from tensorflow import keras

import numpy as np
import tensorflow as tf
import keras

def data_generator():
    for x in range(1, 100):
        yield [np.array(x).astype(np.float32)]

def test_layer(layer):
    model = tf.keras.Model(inputs=[in_tensor], outputs=[layer])
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.representative_dataset = data_generator
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
    ]
    converter.inference_input_type = tf.int16
    converter.inference_output_type = tf.int16
    try:
        tflite_model = converter.convert()
    except Exception as e:
        print("Received exception:\n%s" % str(e))

in_tensor = tf.keras.Input(shape=(1,))
test_layer(tf.math.log(in_tensor))
test_layer(in_tensor ** 3)
test_layer(in_tensor / in_tensor)