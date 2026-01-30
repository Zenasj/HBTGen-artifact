from tensorflow.keras import models

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