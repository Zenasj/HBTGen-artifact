import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import numpy as np

inp = tf.keras.Input([10,20], batch_size = 1, name = "input_0")
x = tf.keras.layers.LSTM(inp.shape[2],
                             return_sequences = True)(inp)
model_lstm = tf.keras.Model(inputs=inp, outputs=x)

rep_data = tf.data.Dataset.from_tensor_slices(np.float32(np.random.random_sample((10,1,10,20))))

def representative_dataset():
        for data in rep_data:
            yield {
            "input_0": data,
            }

converter = tf.lite.TFLiteConverter.from_keras_model(model_lstm)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
tf.lite.OpsSet.TFLITE_BUILTINS,
#comment line below to run at int 8
tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
]
converter.representative_dataset = representative_dataset

calibrated_model = converter.convert()

interpreter = tf.lite.Interpreter(model_content = calibrated_model)
interpreter.allocate_tensors()

import tensorflow as tf

inp = tf.keras.Input([1, 1], batch_size = 1, name = "input_0")
x = tf.keras.layers.LSTM(inp.shape[2], return_sequences = True)(inp)
model_lstm = tf.keras.Model(inputs=inp, outputs=x)

# convert the model to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model_lstm)
tflite_model = converter.convert()

# allocate tensors
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

import tensorflow as tf
import numpy as np

inp = tf.keras.Input([1, 1], batch_size = 1, name = "input_0")
x = tf.keras.layers.LSTM(inp.shape[2], return_sequences = True)(inp)
model_lstm = tf.keras.Model(inputs=inp, outputs=x)

# pick some representative dataset
rep_data = tf.data.Dataset.from_tensor_slices(np.float32(np.random.random_sample((10,1,1,1))))

def representative_dataset():
    for data in rep_data:
        yield [data]

# convert model to TFLite with representative dataset
converter = tf.lite.TFLiteConverter.from_keras_model(model_lstm)
converter.representative_dataset = representative_dataset
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# allocate tensors
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

tflite_model = converter.convert()

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
]
tflite_model = converter.convert()

EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8

UNIDIRECTIONAL_SEQUENCE_LSTM

EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8

OpsSet

converter.target_spec.supported_types = [tf.int8]

import tensorflow as tf
import numpy as np

inp = tf.keras.Input([1, 1], batch_size = 1, name = "input_0")
x = tf.keras.layers.LSTM(inp.shape[2], return_sequences = True)(inp)
model_lstm = tf.keras.Model(inputs=inp, outputs=x)

# pick some representative dataset
rep_data = tf.data.Dataset.from_tensor_slices(np.float32(np.random.random_sample((10,1,1,1))))

def representative_dataset():
    for data in rep_data:
        yield [data]

# convert model to TFLite with representative dataset
converter = tf.lite.TFLiteConverter.from_keras_model(model_lstm)
converter.representative_dataset = representative_dataset
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.int8]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
]
tflite_model = converter.convert()

# allocate tensors
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

py
import torch
import torchvision
import ai_edge_torch

rnn = torch.nn.LSTM(10, 20, 2)
sample_inputs = (torch.randn(5, 3, 10),)

edge_model = ai_edge_torch.convert(rnn.eval(), sample_inputs)
edge_model.export("rnn.tflite")