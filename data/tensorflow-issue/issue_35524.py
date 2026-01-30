from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import os
import tensorflow as tf
import gc # garbage collector
import objgraph
from memory_profiler import profile

def mem_stat():
  objs = gc.get_objects()
  print("total objects count", len(objs))

@profile
def profile_own_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    # model.save('my_model')
    tf.keras.backend.clear_session()
    del model
    gc.collect()

@profile
def profile_load_model(path):
    model = tf.keras.models.load_model(model_path, compile=False)
    tf.keras.backend.clear_session()
    del model
    gc.collect()



model_path = f'/my_model.hd5'
print("load model in loops:")

c = 1
while True:
    print("----------- iter", c)
    profile_load_model(model_path)

    print("mem stat after model creation:")
    mem_stat()
    objgraph.show_growth(limit=30)
    c += 1

import os
import tensorflow as tf
import gc # garbage collector
import objgraph
#from memory_profiler import profile

def mem_stat():
    objs = gc.get_objects()
    print("total objects count", len(objs))

#@profile
def profile_own_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    # model.save('my_model')
    tf.keras.backend.clear_session()
    del model
    gc.collect()

#@profile
def profile_load_model(path):
    model = tf.keras.models.load_model(model_path, compile=False)
    tf.keras.backend.clear_session()
    del model
    gc.collect()

model_path = f'/my_model.hd5'
print("load model in loops:")

c = 1
while True:
    print("----------- iter", c)
    profile_load_model(model_path)

    print("mem stat after model creation:")
    mem_stat()
    objgraph.show_growth(limit=30)
    c += 1

import os
import tensorflow as tf
import gc # garbage collector

def build_and_save_own_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.save('my_model')
    tf.keras.backend.clear_session()
    del model
    gc.collect()

def profile_load_model(path):
    model = tf.keras.models.load_model(model_path, compile=False)
    tf.keras.backend.clear_session()
    del model
    gc.collect()

model_path = 'my_model'
build_and_save_own_model()
print("load model in loops:")
c = 1
while True:
    print("----------- iter", c)
    profile_load_model(model_path)
    c += 1