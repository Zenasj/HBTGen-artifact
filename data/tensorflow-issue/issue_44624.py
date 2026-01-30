from tensorflow.keras import layers
from tensorflow.keras import models

import os
import time
import shutil

import tensorflow as tf
from tensorflow import keras

assert len(tf.config.list_physical_devices('GPU'))>0, "You have to run this on a GPU machine, otherwise you do not see the effect."
print(tf.config.list_physical_devices('GPU'))

model_save_path='tmp'

lstm_layer = keras.layers.LSTM(64, input_shape=(None, 28))
model = keras.models.Sequential([lstm_layer, keras.layers.Dense(10)])

mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer="sgd", metrics=["accuracy"])
model.fit(x_train[0:128], y_train[0:128], validation_data=(x_test, y_test), batch_size=64, epochs=1)
tf.saved_model.save(model, model_save_path)

# Keras load + inference
model_reload_keras = tf.keras.models.load_model(model_save_path)
inp = tf.ones(shape=[8, 512, 28])
for i in range(501):
    # Avoid load offset.
    if i == 1:
        startT = int(round(time.time() * 1000))
    model_reload_keras.predict_on_batch(inp)
endT = int(round(time.time() * 1000))
print('Keras load -- inference time: ' + str(endT-startT) + 'ms')

# TF load + inference
model_reload_tf = tf.saved_model.load(str(model_save_path))
infer = model_reload_tf.signatures['serving_default']
for i in range(501):
    # Avoid load offset.
    if i == 1:
        startT = int(round(time.time() * 1000))
    infer(inp)
endT = int(round(time.time() * 1000))
print('TF load -- inference time (should be as fast as previous run [on GPU]): ' + str(endT - startT) + 'ms')

shutil.rmtree(model_save_path)