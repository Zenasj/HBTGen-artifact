from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import tensorflow as tf

inp = tf.keras.Input((5,16), dtype='float', name='bb')
net = inp[:,3:,:]
net = tf.keras.layers.Conv1D(2,1)(net)

model = tf.keras.Model(inputs=inp, outputs=net)

optimizer = tf.keras.optimizers.Adam()
loss = tf.keras.losses.BinaryCrossentropy()
model.compile(optimizer=optimizer, loss=loss)

model.save('out')

loaded = tf.keras.models.load_model('out')