from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import numpy as np
import sys

print(tf.version.GIT_VERSION, tf.version.VERSION) # v2.4.0-49-g85c8b2a817f 2.4.1
print(sys.version) # 3.8.5 (default, Jul 28 2020, 12:59:40) [GCC 9.3.0]
x = np.dtype('float32').type(3.0)

sigmoid = tf.keras.layers.Activation(activation="sigmoid")
actX = sigmoid(x)

print(type(x)) # <class 'numpy.float32'>
print(type(actX)) # <class 'tensorflow.python.framework.ops.EagerTensor'>

if hasattr(actX, "__len__"):
    print(len(actX))
else:
    print("Has no length")

x = tf.constant(1)
y = tf.constant([1,2,3])

print(x.shape) # ()
print(y.shape) # (3,)