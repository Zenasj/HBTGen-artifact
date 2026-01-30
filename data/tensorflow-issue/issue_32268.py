from tensorflow import keras
from tensorflow.keras import layers

# load tensorflow
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
print(tf.__version__)
AUTOTUNE = tf.data.experimental.AUTOTUNE
#enable eager
tf.enable_eager_execution()
assert tf.multiply(6, 7).numpy() == 42
print("Eager execution: {}".format(tf.executing_eagerly()))

# build our model and print its outputs and summary
base_model = tf.keras.applications.NASNetMobile(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False
output = tf.keras.layers.Dense(num_labels, activation = 'sigmoid', name="cinemanet_output", input_shape=(None, 1056))

model = tf.keras.Sequential([
  base_model,
  tf.keras.layers.GlobalAveragePooling2D(),
  output])
model.summary()
print(model.input.op.name)
print(model.output.op.name)