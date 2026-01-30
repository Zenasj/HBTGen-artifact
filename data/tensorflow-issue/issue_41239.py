from tensorflow import keras
from tensorflow.keras import layers

python
import tensorflow as tf
import numpy as np

DO_BUG = True

inputs = tf.keras.Input((1,))
outputs = tf.keras.layers.Dense(10)(inputs)
model0 = tf.keras.Model(inputs=inputs, outputs=outputs)

if DO_BUG:
    with tf.Graph().as_default():
        inputs = tf.keras.Input((1,))
        outputs = tf.keras.layers.Dense(10)(inputs)
        model1 = tf.keras.Model(inputs=inputs, outputs=outputs)

model0.compile(optimizer=tf.optimizers.SGD(0.1), loss=tf.losses.mse)
model0.fit(np.zeros((4, 1)), np.zeros((4, 10)))

python
import tensorflow as tf
import numpy as np

DO_BUG = True

inputs = tf.keras.Input((1,))
outputs = tf.keras.layers.Dense(10)(inputs)
model0 = tf.keras.Model(inputs=inputs, outputs=outputs)

if DO_BUG:
    with tf.Graph().as_default():
        inputs = tf.keras.Input((1,))
        outputs = tf.keras.layers.Dense(10)(inputs)
        model1 = tf.keras.Model(inputs=inputs, outputs=outputs)

model0.predict(np.zeros((4, 1)))