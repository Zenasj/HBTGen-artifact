from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf

def block(x):
    x =tf.keras.layers.Conv2D(filters=96, kernel_size=3, strides=2, padding="same")(x)
    x =tf.keras.layers.Conv2D(filters=96, kernel_size=3, strides=2, padding="same")(x)
    x =tf.keras.layers.Conv2D(filters=96, kernel_size=3, strides=2, padding="same")(x)
    x =tf.keras.layers.Conv2D(filters=96, kernel_size=3, strides=2, padding="same")(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(10)(x)
    x = tf.keras.layers.Activation(activation=tf.nn.softmax)(x)

    return x
def FCN():
    in_tensor = tf.keras.Input(shape=(32, 32, 3))
    out_tensors = block(in_tensor)
    model = tf.keras.Model(inputs=in_tensor, outputs=out_tensors, name="FCN")
    return model

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf

from test_net import FCN

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# load data and norm to [-1,1]
cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = (x_train - 128.0) / 128.0, (x_test - 128.0) / 128.0

x_test= x_test[0:9984]
y_test = y_test[0:9984]

# config the used GPU memory size
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.3
tf.keras.backend.set_session(tf.Session(config=config))


mirror_strategy = tf.distribute.MirroredStrategy()
with mirror_strategy.scope():
    model = FCN()
    sgd = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=1e-4)
    model.compile(
    optimizer=sgd, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    model.fit(
        x_train, 
        y_train,
        batch_size=256,
        epochs=10,
        validation_data=(x_test, y_test),
        verbose=2,
    )