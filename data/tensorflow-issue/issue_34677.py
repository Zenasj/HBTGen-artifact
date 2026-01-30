from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
import gc
import objgraph

def mem_stat():
  objs = gc.get_objects()
  print("total objects count", len(objs))

c = 1
while True:
  print("----------- iter", c)
  model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
  ])
  
  gc.collect()
  
  print("mem stat after model creation:")
  mem_stat()
  objgraph.show_growth(limit=30)
  c += 1

import tensorflow as tf
import gc
import objgraph


def mem_stat():
  objs = gc.get_objects()
  print("total objects count", len(objs))


c = 1
while True:
  print("----------- iter", c)
  model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(10, activation='softmax')
  ])
  tf.keras.backend.clear_session()
  gc.collect()
  
  print("mem stat after model creation:")
  mem_stat()
  objgraph.show_growth(limit=30)
  c += 1