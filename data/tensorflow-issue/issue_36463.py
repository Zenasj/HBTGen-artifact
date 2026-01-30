from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

seq = tf.keras.Sequential(layers=[
    tf.keras.layers.Dense(units=10, name="d1"),
    tf.keras.layers.Dense(units=20, name="d2"),
])

with tf.name_scope("a"):
    with tf.name_scope("b"):
        seq.build(input_shape=[32, 784])

for w in seq.weights:
    print(w.name)

@tf.function
def func():
    with tf.name_scope("a"):
        with tf.name_scope("b"):
            seq.build(input_shape=[32, 784])

func()

for w in seq.weights:
    print(w.name)

import tensorflow as tf

d = tf.keras.layers.Dense(units=10, name="d1")

with tf.name_scope("a"):
    with tf.name_scope("b"):
        d.build(input_shape=[32, 784])

for w in d.weights:
    print(w.name)