import random
from tensorflow import keras
from tensorflow.keras import layers

tf.saved_model.save()

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Bidirectional, GRU

class SubClassedModel(tf.keras.Model):
    def __init__(self):
        super(SubClassedModel, self).__init__()
        self.model = Bidirectional(GRU(4, time_major=True, return_sequences=True), name='BILSTM_OUTPUT')

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, None, 256], dtype=tf.float32)])
    def call(self, inputs):
        return self.model(inputs)

def convert(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model) 
    return converter.convert()

model = SubClassedModel()
test_input = tf.random.uniform(shape=[2, 2, 256], dtype=tf.float32)
with tf.device('/cpu:0'):
    test_output = model(test_input)

tflite = convert(model)