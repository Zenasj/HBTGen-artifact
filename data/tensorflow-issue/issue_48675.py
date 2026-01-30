import random
from tensorflow import keras

"""
File to show the runtime error when trying to convert MobileNetV3 to TFLite.
"""

import tensorflow as tf


mobile_net = tf.keras.applications.MobileNetV3Large(
    input_tensor=tf.keras.Input(shape=(224, 224, 3), batch_size=1, name="input", dtype=tf.float32)
)
model = tf.keras.Model(inputs=mobile_net.inputs, outputs=mobile_net(mobile_net.inputs))
model.compile()
model.summary(line_length=200)


def representative_dataset_generator():
    """Dataset generator that generates random tensor with the same shape as the input"""
    for _ in range(100):
        yield tf.random.uniform(shape=(1, 224, 224, 3), dtype=tf.float32)


converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Set the representative dataset in order to quantize the activations
converter.representative_dataset = representative_dataset_generator

converter.optimizations = [tf.lite.Optimize.DEFAULT]

converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.target_spec.supported_types = [tf.int8]

# Set the input and output tensors to uint8
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

# Experimental environment
converter.experimental_new_converter = True
converter.experimental_new_quantizer = True

tflite_model = converter.convert()