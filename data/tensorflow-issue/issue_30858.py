from tensorflow import keras

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Linear(layers.Layer):

  def __init__(self, units=32, input_dim=32):
    super(Linear, self).__init__()
    self.w = self.add_weight(shape=(input_dim, units),
                             initializer='random_normal',
                             trainable=True, name="W")
    self.b = self.add_weight(shape=(units,),
                             initializer='zeros',
                             trainable=True, name='b')

  def call(self, inputs):
    return tf.matmul(inputs, self.w) + self.b

class NestedLinear(layers.Layer):

  def __init__(self):
    super(NestedLinear, self).__init__()
    self.linear_1 = Linear(32)

  def call(self, inputs):
    x = self.linear_1(inputs)
    return x

inputs = layers.Input(shape=(32,))

print("SAVING SINGLE LAYER")
outputs = Linear(32)(inputs)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
for var in model.trainable_variables:
    print(var)
ckpt = tf.train.Checkpoint(model=model)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
with sess.as_default():
    ckpt.save("ckpt")

for var in tf.train.list_variables("./"):
    print(var)

print("--------------------------------------")

nl_layer = NestedLinear()
outputs = nl_layer(inputs)

print("SAVING NESTED LAYER")

model = tf.keras.Model(inputs=inputs, outputs=outputs)
for var in model.trainable_variables:
    print(var)
ckpt = tf.train.Checkpoint(model=model)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
with sess.as_default():
    ckpt.save("ckpt")

for var in tf.train.list_variables("./"):
    print(var)