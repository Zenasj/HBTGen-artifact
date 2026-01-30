import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

NUM_SAMPLES = 10
SAMPLE_SIZE = 64

def convert(model: tf.keras.Model, dataset_gen):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()

def gru_dataset():
    for _ in range(100):
        shape = (1, NUM_SAMPLES, SAMPLE_SIZE)
        yield [tf.random.uniform(shape, minval=-1, maxval=1)]

in_layer = tf.keras.layers.Input((NUM_SAMPLES, SAMPLE_SIZE), batch_size=1)
out_layer = tf.keras.layers.GRU(units=16)(in_layer)
gru_model = tf.keras.Model(in_layer, out_layer)
int8_model = convert(gru_model, gru_dataset)