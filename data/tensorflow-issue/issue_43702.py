from tensorflow.keras import layers

import gc
import os

import numpy as np
import psutil
import tensorflow as tf

tf.get_logger().setLevel("ERROR")  # Suppress "tf.function retracing" warnings
process = psutil.Process(os.getpid())
for i in range(100):
    # do some work
    model = tf.keras.applications.mobilenet.MobileNet()
    model.compile(loss="mse")
    x = tf.zeros((1, *model.input.shape[1:]))
    y = tf.zeros((1, *model.output.shape[1:]))
    history = model.fit(x=x, y=y, verbose=0)
    
    # clean up
    _ = gc.collect()
    tf.keras.backend.clear_session()
    
    # show memory usage
    print(f"iteration {i}: rss {process.memory_info().rss >> 20} MB")

import gc
import os

import psutil
import tensorflow as tf
from tensorflow import keras

tf.get_logger().setLevel("ERROR")  # Suppress "tf.function retracing" warnings
process = psutil.Process(os.getpid())
prev_mem = 0
first_mem = 0
for i in range(100):
    # do some work
    in_layer = keras.layers.Input(shape=(1,))
    hidden_layer = in_layer
    for _ in range(200):
        hidden_layer = keras.layers.Dense(1)(hidden_layer)
    out_layer = hidden_layer
    model = keras.Model(inputs=in_layer, outputs=out_layer)
    model.compile(loss="mse")
    history = model.fit(x=[0], y=[0], verbose=0)
    
    # clean up
    _ = gc.collect()
    keras.backend.clear_session()
    
    # show memory usage
    mem = process.memory_info().rss
    if i == 0:
        first_mem = mem
    print(
        f"iteration {i}: rss {mem >> 20} MB ({(mem - prev_mem) >> 10:+} KB; "
        + f"{((mem - first_mem) // max(1, i)) >> 10:+} KB/it.)"
    )
    prev_mem = mem

import gc
import os

import psutil
import tensorflow as tf
from tensorflow import keras


def do_some_work():
    in_layer = keras.layers.Input(shape=(1,))
    hidden_layer = in_layer
    for _ in range(200):
        hidden_layer = keras.layers.Dense(1)(hidden_layer)
    out_layer = hidden_layer
    model = keras.Model(inputs=in_layer, outputs=out_layer)
    model.compile(loss="mse", run_eagerly=True)
    history = model.fit(x=[0], y=[0], verbose=0)


print(tf.__version__)
tf.get_logger().setLevel("ERROR")  # Suppress "tf.function retracing" warnings
process = psutil.Process(os.getpid())
prev_mem = 0
first_mem = 0
for i in range(1000):
    do_some_work()

    # clean up
    keras.backend.clear_session()
    _ = gc.collect()
    
    # show memory usage
    mem = process.memory_info().rss
    if i == 0:
        first_mem = mem
    print(
        f"iteration {i}: rss {mem >> 20} MB ({(mem - prev_mem) >> 10:+} KB; "
        + f"{((mem - first_mem) // max(1, i)) >> 10:+} KB/it.)"
    )
    prev_mem = mem