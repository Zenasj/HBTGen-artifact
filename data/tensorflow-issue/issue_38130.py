import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

def loss_func(model, x, y):
    y_ = model(x)
    return tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.AUTO)(y_true=y, y_pred=y_)

def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss_func(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

class ScoMatLayer(tf.keras.layers.Layer):
    def __init__(self, embd_layer):
        super(ScoMatLayer, self).__init__()
        all_index = tf.range(nb_item, dtype=tf.int32)
        self.embd2 = embd_layer(all_index)
    def call(self, inputs):
        return tf.matmul(inputs, self.embd2, transpose_b=True)

batch_size = 2
nb_item = 5
nb_hidden = 3

inputs = np.array([[1], [2]])
targets = np.random.randn(batch_size, nb_item)

input_layer = tf.keras.layers.Input((1, ), dtype=tf.int32)

embd_layer = tf.keras.layers.Embedding(nb_item, nb_hidden)

embd1 = tf.reshape(embd_layer(input_layer), [-1, nb_hidden])

scl = ScoMatLayer(embd_layer)

sco_mat = scl(embd1)

model = tf.keras.models.Model(inputs=input_layer, outputs=[sco_mat])

loss, grads = grad(model, inputs, targets)
print(grads[0].values.numpy())