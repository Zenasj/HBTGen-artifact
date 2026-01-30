from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf

class CondModel(tf.keras.models.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense_layer = tf.keras.layers.Dense(units=1)

    def call(self, inputs, training=None):
        if training:
            return self.dense_layer(inputs)
        else:
            return self.dense_layer(inputs)


m = CondModel()
m.__call__ = tf.function(m.__call__)
m.__call__.get_concrete_function(
    inputs=tf.TensorSpec(shape=[1, 1]), 
    training=tf.TensorSpec(shape=None, dtype=tf.bool)
)
tf.saved_model.save(m, 'saved_model')

import tensorflow as tf

reloaded = tf.saved_model.load('saved_model')

with tf.GradientTape() as tape:
    reloaded(inputs=[[1]], training=False)

import tensorflow as tf

# --- HACK ---
def f(x):
    if x:
        pass 
    
tf.function(f).get_concrete_function(
    x=tf.TensorSpec(shape=None)
)
# -------------

reloaded = tf.saved_model.load('saved_model')

with tf.GradientTape() as tape:
    reloaded(inputs=[[1]], training=False)

import tensorflow as tf

reloaded = tf.saved_model.load('saved_model')

with tf.GradientTape() as tape:
    reloaded(inputs=[[1]], training=False)