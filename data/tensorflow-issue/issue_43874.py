import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

features =  tf.random.normal(shape=(100, 1, 10))
labels = tf.random.normal((100,1,1))
dataset = tf.data.Dataset.from_tensor_slices((features,labels))
ds_iter = iter(dataset) # tensorflow.python.data.ops.iterator_ops.OwnedIterator
features.shape, labels.shape

x = tf.keras.layers.Input(shape=[10])
y_pred = tf.keras.layers.Dense(1, activation='sigmoid', name="L0")(x)
model = tf.keras.Model(x, y_pred)
model.compile(optimizer='sgd', loss='mse',)
model.fit(ds_iter, epochs=1)