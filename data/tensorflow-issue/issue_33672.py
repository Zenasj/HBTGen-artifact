from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.python.client import device_lib

print("TensorFlow version is", tf.__version__)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
NUM_SAMPLES = 1000
x_train = x_train[:NUM_SAMPLES]
y_train = y_train[:NUM_SAMPLES]
x_test = x_test[:NUM_SAMPLES]
y_test = y_test[:NUM_SAMPLES]

def fake3d(x):
    return np.repeat(x[:, np.newaxis], 8, axis=1)

x_train = fake3d(x_train)
x_test = fake3d(x_test)

num_classes = np.max(y_train) + 1
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

def normalize(ndarray):
    ndarray = ndarray.astype("float32")
    ndarray = ndarray/255.0
    return ndarray

x_train = normalize(x_train)
x_test = normalize(x_test)

def create_model(num_classes=10):
    # model parameters
    act = "relu"
    pad = "same"
    ini = "he_uniform"

    model = tf.keras.models.Sequential([
        Conv3D(128, (3,3,3), activation=act, padding=pad, kernel_initializer=ini,
               input_shape=(8,32,32,3)),
        Conv3D(256, (3,3,3), activation=act, padding=pad, kernel_initializer=ini),
        Conv3D(256, (3,3,3), activation=act, padding=pad, kernel_initializer=ini),
        Conv3D(256, (3,3,3), activation=act, padding=pad, kernel_initializer=ini),
        MaxPooling3D(pool_size=(2,2,2)),
        Conv3D(256, (3,3,3), activation=act, padding=pad, kernel_initializer=ini),
        Conv3D(256, (3,3,3), activation=act, padding=pad, kernel_initializer=ini),
        Conv3D(512, (3,3,3), activation=act, padding=pad, kernel_initializer=ini),
        Conv3D(512, (3,3,3), activation=act, padding=pad, kernel_initializer=ini),
        MaxPooling3D(pool_size=(2,2,2)),
        Conv3D(256, (3,3,3), activation=act, padding=pad, kernel_initializer=ini),
        Conv3D(256, (3,3,3), activation=act, padding=pad, kernel_initializer=ini),
        Conv3D(256, (3,3,3), activation=act, padding=pad, kernel_initializer=ini),
        Conv3D(128, (3,3,3), activation=act, padding=pad, kernel_initializer=ini),
        MaxPooling3D(pool_size=(2,4,4)),
        Flatten(),
        BatchNormalization(),
        Dense(512, activation='relu'),
        Dense(num_classes, activation="softmax")
    ])

    return model

model = create_model(num_classes)
model.summary()
BATCH_SIZE = 320
N_EPOCHS = 6
opt = tf.keras.optimizers.SGD(learning_rate=0.02, momentum=0.5)

def train_model(mixed_precision, optimizer):
    model = create_model(num_classes)

    if mixed_precision:
        import tensorflow
        optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)

    model.compile(loss="categorical_crossentropy",
                  optimizer=optimizer,
                  metrics=["accuracy"])

    train_start = time.time()

    train_log = model.fit(x_train, y_train,
                          batch_size=BATCH_SIZE,
                          epochs=N_EPOCHS,
                          use_multiprocessing=True,
                          workers=2)

    train_end = time.time()

    results = {
               "train_time": train_end-train_start,
               "train_log": train_log}

    return results

fp32_results = train_model(mixed_precision=False, optimizer=opt)
train_time = round(fp32_results["train_time"], 1)
print("achieved in", train_time, "seconds")

tf.keras.backend.clear_session()
time.sleep(10)

mp_results = train_model(mixed_precision=True, optimizer=opt)
train_time = round(mp_results["train_time"], 1)
print("achieved in", train_time, "seconds")

TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_WHITELIST_ADD=Conv3D,Conv3DBackpropFilter,Conv3DBackpropFilterV2,Conv3DBackpropInput,Conv3DBackpropInputV2

import os
os.environ['TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_WHITELIST_ADD'] = 'Conv3D,Conv3DBackpropFilter,Conv3DBackpropFilterV2,Conv3DBackpropInput,Conv3DBackpropInputV2'