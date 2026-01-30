import random
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import tensorflow as tf
#tf.enable_eager_execution()

train_X = np.arange(1, 1001).reshape((200, 5))
train_Y = np.array(list(map(
    lambda x: np.array([1, 0]) if x == 0 else np.array([0, 1]),
    np.random.randint(2, size=200))))

ds = tf.data.Dataset.from_tensor_slices((train_X, train_Y))

ws = 3
sh = None
st = 1
bs = 10
nu = 86
ne = 5

ds = ds.window(size=ws, shift=sh, stride=st, drop_remainder=True).flat_map(lambda x, y: tf.data.Dataset.zip((x.batch(ws), y.batch(ws)))).batch(bs, drop_remainder=True)

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(ws, len(train_X[0])), batch_size=bs),
    tf.keras.layers.LSTM(nu, return_sequences=True, stateful=True),
    tf.keras.layers.Dense(2), # categorical
    tf.keras.layers.Activation('softmax'), # categorical
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
print(model.summary())

history = {}
for e in range(ne):
    history[e] = model.fit(ds, epochs=1, shuffle=False)
    model.reset_states()

#history = model.fit(ds, epochs=ne, shuffle=False)