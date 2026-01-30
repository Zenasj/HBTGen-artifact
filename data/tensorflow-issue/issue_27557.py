import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import scipy.sparse
import numpy as np

def input_fn():

    x = scipy.sparse.random(1, 400)
    y = scipy.random.randint(2, size=(1,1))

    indices = np.mat([x.row, x.col]).transpose()
    sp = tf.sparse.SparseTensor(indices, x.data, x.shape)
    d = tf.data.Dataset.from_tensors((sp,y))
    return d

input_layer = tf.keras.layers.Input(shape=(400, ), sparse=True)
weights = tf.get_variable(name='weights', shape=(400, 1))

weights_mult = lambda x: tf.sparse_tensor_dense_matmul(x, weights)
output_layer=  tf.keras.layers.Lambda(weights_mult)(input_layer)
model = tf.keras.Model([input_layer], output_layer)
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
d = input_fn()
model.fit(d.make_one_shot_iterator(), epochs=3, steps_per_epoch=1)