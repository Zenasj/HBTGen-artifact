import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
import numpy as np
from tqdm import tqdm
from memory_profiler import profile

data_array = np.random.random_sample((1, 1024))
tf_array = tf.constant(data_array, dtype=tf.float32)

input = tf.keras.Input((1, 1024))
hidden_layer = tf.keras.layers.Dense(1024)(input)
output = tf.keras.layers.Dense(1)(hidden_layer)
model = tf.keras.Model(inputs=[input], outputs=[output])

pred = model([tf_array])
print(pred)


@profile
def func():
    export_path = "temp_export"
    tf.saved_model.save(model, export_path)
    imported = tf.saved_model.load(export_path)


for i in tqdm(range(1000000), total=1000000):
    func()

tf-nightly-gpu

import tensorflow as tf
import psutil
import gc

input = tf.keras.Input((1, 1024))
dense1 = tf.keras.layers.Dense(1024)(input)
dense2 = tf.keras.layers.Dense(1024)(dense1)
dense2 = tf.keras.layers.BatchNormalization()(dense2)
dense2 = tf.keras.layers.LeakyReLU()(dense2)
output = tf.keras.layers.Dense(1)(dense2)
model = tf.keras.Model(inputs=[input], outputs=[output])

def func():
  export_path = "temp_export.h5"
  model.save(export_path)
  tf.keras.models.load_model(export_path)
  tf.keras.backend.clear_session()

for i in range(1000000):
    func()
    if i % 100 == 0:
        print(i, ": free memory", psutil.virtual_memory().available / (1024.0 ** 2), "Mb") 
    gc.collect()

import gc

import psutil
import tensorflow as tf
from tqdm import tqdm

input = tf.keras.Input((1, 1024))
dense1 = tf.keras.layers.Dense(1024)(input)
dense2 = tf.keras.layers.Dense(1024)(dense1)
dense2 = tf.keras.layers.BatchNormalization()(dense2)
dense2 = tf.keras.layers.LeakyReLU()(dense2)
output = tf.keras.layers.Dense(1)(dense2)
model = tf.keras.Model(inputs=[input], outputs=[output])


def func():
    export_path = "temp_export.h5"
    model.save(export_path)
    tf.keras.backend.clear_session()
    gc.collect()


first_memory_usage = psutil.virtual_memory().available
progress_bar = tqdm()
step = 0
while True:
    func()
    progress_bar.set_description(
        f"Already lost {(first_memory_usage - psutil.virtual_memory().available) / (1024 ** 2):.3f} MB from tf.keras.Model.save"
    )
    step += 1
    progress_bar.update(step)

import gc

import psutil
import tensorflow as tf
from tqdm import tqdm

if __name__ == "__main__":

    input = tf.keras.Input((1, 1024))
    dense1 = tf.keras.layers.Dense(1024)(input)
    dense2 = tf.keras.layers.Dense(1024)(dense1)
    dense2 = tf.keras.layers.BatchNormalization()(dense2)
    dense2 = tf.keras.layers.LeakyReLU()(dense2)
    output = tf.keras.layers.Dense(1)(dense2)
    model = tf.keras.Model(inputs=[input], outputs=[output])

    first_memory_usage = psutil.virtual_memory().available
    progress_bar = tqdm()
    step = 0
    while True:
        model.save_weights("test.h5")

        tf.keras.backend.clear_session()
        gc.collect()
        progress_bar.set_description(
            f"Already lost {(first_memory_usage - psutil.virtual_memory().available) / (1024 ** 2):.3f} MB from tf.keras.Model.save_weights"
        )
        step += 1
        progress_bar.update(step)