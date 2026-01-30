from tensorflow.keras import models

def MyModel_keras():
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(conf.n_hidden_lstm, activation='tanh', return_sequences=False, name='lstm1'),
        tf.keras.layers.Dense(conf.n_dense_1, activation='relu', name='dense1'),
        tf.keras.layers.Dense(conf.num_output_classes, activation='softmax', name='dense2')
    ])
    return model

from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.datasets import cifar100
import numpy as np
import tensorflow as tf
from tensorflow import keras
keras.backend.clear_session()
keras.backend.set_learning_phase(0)
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
x_train = np.float32(x_train)[:64*(x_train.shape[0]//64)]
x_train /= 255.0
i_shape = x_train[0].shape

inputs = layers.Input(i_shape)
base_model = keras.models.Sequential([
    layers.Conv2D(128, 3, padding='same', strides=(2, 2)),
    layers.BatchNormalization(),
    layers.LeakyReLU(0.2),

    layers.Conv2D(256, 3, padding='same', strides=(2, 2)),
    layers.BatchNormalization(),
    layers.LeakyReLU(0.2),

    layers.Conv2D(512*100, 3, padding='same', strides=(2, 2)),
    layers.BatchNormalization(),
    layers.LeakyReLU(0.2),
    layers.Flatten(),
    layers.Dense(5*128),
    layers.Reshape((5, 128)),
    layers.LSTM(128),
    layers.Flatten(),
])(inputs)

prediction = layers.Dense(100, activation='softmax')(base_model)
model = keras.Model(inputs=inputs, outputs=prediction)
optimizer = optimizers.Adam(0.00001)
model.compile(optimizer, 'categorical_crossentropy', metrics=['accuracy'])
print(model.summary())
saved_model_dir = os.getcwd()
output_directory = os.getcwd()
tf.saved_model.save(model, saved_model_dir)

# graph quantization
loaded = tf.saved_model.load(saved_model_dir)

params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
    precision_mode='FP16',
    is_dynamic_op=True,
    maximum_cached_engines=16)
converter = trt.TrtGraphConverterV2(
    input_saved_model_dir=saved_model_dir,
    input_saved_model_tags="serve",
    input_saved_model_signature_key="serving_default",
    conversion_params=params)
converter.convert()
saved_model_dir_trt = output_directory
converter.save(saved_model_dir_trt)

import tensorflow as tf

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, embedding_dim]),
    tf.keras.layers.LSTM(rnn_units,
                        return_sequences=True,
                        stateful=False,
                        recurrent_activation='sigmoid',
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model

embedding_dim = 100
units = 256
vocab_size = 300
batch_size = 32

model = build_model(vocab_size, embedding_dim, units, batch_size)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

from tensorflow.python.keras.saving import saving_utils as _saving_utils
from tensorflow.python.framework import convert_to_constants as _convert_to_constants

tf.keras.backend.set_learning_phase(False)
func = _saving_utils.trace_model_call(model)
concrete_func = func.get_concrete_function()
frozen_func = _convert_to_constants.convert_variables_to_constants_v2(concrete_func)