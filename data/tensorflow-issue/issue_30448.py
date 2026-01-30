import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import numpy as np
import time

SIZE = 5000

inp = tf.keras.layers.Input(shape=(SIZE,), dtype='float32')
x = tf.keras.layers.Dense(units=SIZE)(inp)

model = tf.keras.Model(inputs=inp, outputs=x)

np_data = np.random.rand(1, SIZE)
ds = tf.data.Dataset.from_tensor_slices(np_data).batch(1).repeat()

debug_time = time.time()
while True:
    model.predict(x=ds, steps=1)
    print('Processing time {:.2f}'.format(time.time() - debug_time))
    debug_time = time.time()

import tensorflow as tf
import numpy as np
import time

SIZE = 5000

inp = tf.keras.layers.Input(shape=(SIZE,), dtype='float32')
x = tf.keras.layers.Dense(units=SIZE)(inp)

model = tf.keras.Model(inputs=inp, outputs=x)

np_data = np.random.rand(1, SIZE)

debug_time = time.time()
while True:
    model.predict(x=np_data)  # using numpy array directly
    print('Processing time {:.2f}'.format(time.time() - debug_time))
    debug_time = time.time()