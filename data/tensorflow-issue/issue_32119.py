from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

inputs = Input(shape = [1], dtype = tf.float32)
labels = Input(shape = [1], dtype = tf.float32)
outputs = Dense(1, use_bias = True, activation = None)(inputs)

model = Model([inputs, labels], outputs)
loss = tf.square(labels - outputs)
model.add_loss(loss)

model.compile(Adam(0.1))

rg = tf.data.Dataset.range(128)
rg = rg.map(lambda x: tf.cast(x, tf.float32))

'''
# This snippet works fine
ds2 = rg.map(lambda x: ((x, 2*x+1), 0))
ds2 = ds2.batch(16)
model.fit(ds2, epochs = 50)
print(model.predict(([-32], [0])))
'''

# This snippet causes exception
ds3 = rg.map(lambda x: ((x, 2*x+1), 0, 1))
ds3 = ds3.batch(16)
model.fit(ds3, epochs = 50)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

inputs = Input(shape = [1], dtype = tf.float32)
labels = Input(shape = [1], dtype = tf.float32)
outputs = Dense(1, use_bias = True, activation = None)(inputs)

model = Model([inputs, labels], outputs)
loss = tf.square(labels - outputs)
model.add_loss(loss)

model.compile(Adam(0.1))

rise = tf.data.Dataset.range(60)
rise = rise.map(lambda x: tf.cast(x, tf.float32))
rise = rise.map(lambda x: ((x, x), [0, 1]))

fall = tf.data.Dataset.range(4)
fall = fall.map(lambda x: tf.cast(x, tf.float32))
fall = fall.map(lambda x: ((x, -x), [0, 9999999]))

ds = rise.concatenate(fall)
ds = ds.batch(16)

model.fit(ds, epochs = 50)
p = model.predict(([-100], [0]))
print(p) # p is around [[-100]], which means that y=x is acturally learned.

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

inputs = Input(shape = [1], dtype = tf.float32)
labels = Input(shape = [1], dtype = tf.float32)
sample_weights = Input(shape = [1], dtype = tf.float32)
outputs = Dense(1, use_bias = True, activation = None)(inputs)

model = Model([inputs, labels, sample_weights], outputs)
loss = tf.reduce_mean(tf.square(labels - outputs) / 2 * sample_weights)
model.add_loss(loss)

model.compile(Adam(0.1))

rise = tf.data.Dataset.range(60)
rise = rise.map(lambda x: tf.cast(x, tf.float32))
rise = rise.map(lambda x: ((x, x, 1), 0))

fall = tf.data.Dataset.range(4)
fall = fall.map(lambda x: tf.cast(x, tf.float32))
fall = fall.map(lambda x: ((x, -x, 9999999), 0))

ds = rise.concatenate(fall)
ds = ds.batch(16)

model.fit(ds, epochs = 50)
p = model.predict(([-100], [0], [1]))
print(p) # p is around [[100]]