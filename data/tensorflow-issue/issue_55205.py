import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import tensorflow_model_optimization as tfmot

inp = tf.keras.Input(shape=(8, 4), batch_size=1)
out = tf.keras.layers.Dense(16)(inp)
model = tf.keras.Model(inp,out)

def representative_dataset():
    for _ in range(100):
        yield [tf.random.uniform(shape=(1, 8, 4))]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_model = converter.convert()
with open('./dense_ptq.tflite', 'wb') as f:
    f.write(tflite_model)

tf.keras.backend.clear_session()

inp = tf.keras.Input(shape=(8, 4), batch_size=1)
out = tfmot.quantization.keras.quantize_annotate_layer(tf.keras.layers.Dense(16))(inp)
model = tfmot.quantization.keras.quantize_apply(tf.keras.Model(inp,out))

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_model = converter.convert()
with open('./dense_qat.tflite', 'wb') as f:
    f.write(tflite_model)