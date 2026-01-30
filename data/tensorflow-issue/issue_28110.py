from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import numpy as np
import tensorflow as tf

# tf.config.gpu.set_per_process_memory_growth(True)

size = 28000

inputs = tf.keras.Input((size,), dtype='float32')
outputs = tf.keras.layers.Dense(size)(inputs)
model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

model.predict(np.ones((1, size,), dtype=np.float32))

print('complete')

while True:
    pass

import keras
import numpy as np

size = 28000

inputs = keras.Input((size,), dtype='float32')
outputs = keras.layers.Dense(size)(inputs)
model = keras.models.Model(inputs=inputs, outputs=outputs)

model.predict(np.ones((1, size,), dtype=np.float32))

print('complete')

while True:
    pass

a = tf.ones((size, size))
b = a + 1
c = b + 1

size = 40000  # Slightly larger b/c I was testing on a GPU w/ 16 GB
init_fn = tf.keras.initializers.glorot_uniform()
init_fn = tf.function(init_fn, autograph=False) # <== This is what prevents the OOM (Comment it out to test)
layer = tf.keras.layers.Dense(size, kernel_initializer=init_fn)
layer.build((size,))