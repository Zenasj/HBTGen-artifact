import random
from tensorflow import keras

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

def dataset_gen():
    for _ in range(10):
        yield [np.random.randint(0,256, [1,300,300,3]).astype(np.uint8)]

inputs = tf.keras.Input(shape=(300,300,3), dtype=tf.uint8)
layers = hub.KerasLayer('https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2')(inputs)
keras_model = tf.keras.Model(inputs=inputs, outputs=layers)

converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS] # [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()

with open('ssd_mobilenet_v2_tfhub_quant.tflite', 'wb') as f:
    f.write(tflite_model)

converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# Set the input and output tensors to uint8 (APIs added in r2.3)
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8