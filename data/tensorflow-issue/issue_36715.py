import random
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import tensorflow as tf
id = np.ones(shape=(2,2))
i = tf.keras.layers.Input(shape=(2,),dtype=tf.float64)
y = tf.random.normal(shape=(2,2), name="noise", dtype=tf.float64)
o = tf.add(i, y)
model = tf.keras.Model(inputs=i, outputs=o)
model.predict(id)

import numpy as np
import tensorflow as tf
x = tf.constant(value=np.ones(shape=(2,2)), dtype=tf.float64)
y = tf.random.normal(shape=(2,2), name="noise", dtype=tf.float64)
z = tf.add(x, y)
print(z)

import tensorflow as tf
y = tf.random.normal(shape=(2,2), name="noise", dtype=tf.float64)
print(y)
print(y)

y = tf.keras.layers.Lambda(lambda _: tf.random.normal(shape=(32,10), name="noise", dtype=tf.float64), dtype=tf.float64)(i)

for sample in my_dataset:
    noise = tf.random.normal()
    output = model(sample + noise)