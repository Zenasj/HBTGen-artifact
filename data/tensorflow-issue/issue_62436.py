import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

alpha = 0.2
__input___0_tensor = tf.random.uniform([2, 1, 1, 1], minval=0, maxval=0,
                                                    dtype=tf.float64)
__input___0 = tf.identity(
    __input___0_tensor)
LeakyReLU_class = tf.keras.layers.LeakyReLU(alpha=alpha, dtype=tf.float64)

layer = LeakyReLU_class
inputs = __input___0

r = LeakyReLU_class(inputs)
theoretical, numerical = tf.test.compute_gradient(LeakyReLU_class, [inputs])
print(theoretical)
print(numerical)

(array([[0.2, 0. ],
       [0. , 0.2]]),)
(array([[0.6, 0. ],
       [0. , 0.6]]),)