import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

# changing this to 4 make train success
n_inputs = 5
dim_embedding = 10

inputs = []
for i in range(n_inputs):
  inputs.append(tf.keras.Input(shape=(1,), dtype='int32'))

embeds = []
shared_embed = tf.keras.layers.Embedding(n_input, dim_embedding)
for i in range(n_inputs):
  embed = shared_embed(inputs[i])
  flatten = tf.keras.layers.Flatten()(embed)
  embeds.append(flatten)

concat = tf.keras.layers.Concatenate(axis=1)(embeds)
output = tf.keras.layers.Dense(1)(concat)
model = tf.keras.Model(inputs=[inputs], outputs=[output])
model.summary()

model.compile(optimizer='adam', loss='mse')

# run
dummy_input = []
for i in range(n_shared_layers):
  dummy_input.append(np.random.randint(0, n_input, size=(1000, )))

dummy = np.arange(1000).reshape(1000, 1, 1)

model.fit(x=dummy_input, y=dummy, batch_size=32, epochs=1, verbose=1)

import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

# changing this to 4 make train success
n_inputs = 5
dim_embedding = 10

inputs = []
for i in range(n_inputs):
  inputs.append(tf.keras.Input(shape=(1,), dtype='int32'))

embeds = []
shared_embed = tf.keras.layers.Embedding(n_input, dim_embedding)
for i in range(n_inputs):
  embed = shared_embed(inputs[i])
  flatten = tf.keras.layers.Flatten()(embed)
  embeds.append(flatten)

concat = tf.keras.layers.Concatenate(axis=1)(embeds)
output = tf.keras.layers.Dense(1)(concat)
model = tf.keras.Model(inputs=[inputs], outputs=[output])
model.summary()

model.compile(optimizer='adam', loss='mse')

# run
dummy_input = []
for i in range(n_shared_layers):
  dummy_input.append(np.random.randint(0, n_input, size=(1000, )))

dummy = np.arange(1000).reshape(1000, 1, 1)

model.fit(x=dummy_input, y=dummy, batch_size=32, epochs=1, verbose=1)

import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

# changing this to 4 make train success
n_inputs = 5
n_items = 1000
dim_embedding = 10

inputs = []
for i in range(n_inputs):
  inputs.append(tf.keras.Input(shape=(1,), dtype='int32'))

embeds = []
shared_embed = tf.keras.layers.Embedding(n_items, dim_embedding)
for i in range(n_inputs):
  embed = shared_embed(inputs[i])
  flatten = tf.keras.layers.Flatten()(embed)
  embeds.append(flatten)

concat = tf.keras.layers.Concatenate(axis=1)(embeds)
output = tf.keras.layers.Dense(1)(concat)
model = tf.keras.Model(inputs=[inputs], outputs=[output])
model.summary()

model.compile(optimizer='adam', loss='mse')

# run
dummy_inputs = []
for i in range(n_inputs):
  dummy_inputs.append(np.random.randint(0, n_inputs, size=(n_items, )))

dummy = np.arange(n_items).reshape(n_items, 1, 1)

model.fit(x=dummy_inputs, y=dummy, batch_size=32, epochs=1, verbose=1)