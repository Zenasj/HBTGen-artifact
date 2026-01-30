import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

with tf.Graph().as_default():  
  inputs = tf.keras.Input(shape=[2])
  with tf.name_scope('block'):
      outputs = tf.keras.layers.Dense(10)(inputs)
model = tf.keras.Model(inputs, outputs)
for w in model.weights:
    print(w.name)