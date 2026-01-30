import random
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import tensorflow as tf
import numpy as np
a = np.array([2 , 4, 5])
ap=tf.constant(a)

import tensorflow as tf
major_version = int(tf.__version__.split(".")[0])
if major_version >= 2:
   from tensorflow.python import _pywrap_util_port
   print("MKL enabled:", _pywrap_util_port.IsMklEnabled())
else:
   print("MKL enabled:", tf.pywrap_tensorflow.IsMklEnabled())

import tensorflow as tf
import numpy as np
a = np.array([2 , 4, 5])
ap=tf.constant(a)

import tensorflow as tf
major_version = int(tf.__version__.split(".")[0])
if major_version >= 2:
   from tensorflow.python import _pywrap_util_port
   print("MKL enabled:", _pywrap_util_port.IsMklEnabled())
else:
   print("MKL enabled:", tf.pywrap_tensorflow.IsMklEnabled())

tf.__version__
'2.3.0'

import tensorflow as tf
major_version = int(tf.__version__.split(".")[0])
if major_version >= 2:
    from tensorflow.python import _pywrap_util_port
    print("MKL enabled:", _pywrap_util_port.IsMklEnabled())
else:
    print("MKL enabled:", tf.pywrap_tensorflow.IsMklEnabled())

import tensorflow as tf
from tensorflow import keras

input_A = keras.layers.Input(shape=[5], name="wide_input")
input_B = keras.layers.Input(shape=[6], name="deep_input")
hidden1 = keras.layers.Dense(30, activation="relu")(input_B)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
concat = keras.layers.concatenate([input_A, hidden2])
output = keras.layers.Dense(1, name="output")(concat)
model = keras.models.Model(inputs=[input_A, input_B], outputs=[output])

import numpy as np
import tensorflow as tf

x_in = np.array([[
  [[2], [1], [2], [0], [1]],
  [[1], [3], [2], [2], [3]],
  [[1], [1], [3], [3], [0]],
  [[2], [2], [0], [1], [1]],
  [[0], [0], [3], [1], [2]], ]])
kernel_in = np.array([
 [ [[2, 0.1]], [[3, 0.2]] ],
 [ [[0, 0.3]],[[1, 0.4]] ], ])
x = tf.constant(x_in, dtype=tf.float32)
kernel = tf.constant(kernel_in, dtype=tf.float32)
tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='VALID')

import tensorflow as tf
major_version = int(tf.__version__.split(".")[0])
if major_version >= 2:
    from tensorflow.python import _pywrap_util_port
    print("MKL enabled:", _pywrap_util_port.IsMklEnabled())
else:
    print("MKL enabled:", tf.pywrap_tensorflow.IsMklEnabled())

set

import mkl
mkl.verbose(1)

import tensorflow as tf
major_version = int(tf.__version__.split(".")[0])
if major_version >= 2:
    from tensorflow.python import _pywrap_util_port
    print("MKL enabled:", _pywrap_util_port.IsMklEnabled())
else:
    print("MKL enabled:", tf.pywrap_tensorflow.IsMklEnabled())

from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional, Conv1D
from tensorflow.keras.layers import Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import numpy as np
from time import time


def timeit(func, iterations, *args):
    t0 = time()
    for _ in range(iterations):
        func(*args)
    print("Time/iter: %.4f sec" % ((time() - t0) / iterations))

def make_small_model(batch_shape):
    ipt   = Input(batch_shape=batch_shape)
    x     = Conv1D(128, 400, strides=4, padding='same')(ipt)
    x     = Flatten()(x)
    x     = Dropout(0.5)(x)
    x     = Dense(64, activation='relu')(x)
    out   = Dense(1,  activation='sigmoid')(x)
    model = Model(ipt, out)
    model.compile(Adam(lr=1e-4), 'binary_crossentropy')
    return model

def make_medium_model(batch_shape):
    ipt   = Input(batch_shape=batch_shape)
    x     = Bidirectional(LSTM(512, activation='relu', return_sequences=True))(ipt)
    x     = LSTM(512, activation='relu', return_sequences=True)(x)
    x     = Conv1D(128, 400, strides=4, padding='same')(x)
    x     = Flatten()(x)
    x     = Dense(256, activation='relu')(x)
    x     = Dropout(0.5)(x)
    x     = Dense(128, activation='relu')(x)
    x     = Dense(64,  activation='relu')(x)
    out   = Dense(1,   activation='sigmoid')(x)
    model = Model(ipt, out)
    model.compile(Adam(lr=1e-4), 'binary_crossentropy')
    return model

def make_data(batch_shape):
    return np.random.randn(*batch_shape), np.random.randint(0, 2, (batch_shape[0], 1))

batch_shape = (32, 400, 16)
X, y = make_data(batch_shape)

model_small = make_small_model(batch_shape)
model_small.train_on_batch(X, y)  # skip first iteration which builds graph
timeit(model_small.train_on_batch, 200, X, y)

K.clear_session()  # in my testing, kernel was restarted instead

model_medium = make_medium_model(batch_shape)
model_medium.train_on_batch(X, y)  # skip first iteration which builds graph
timeit(model_medium.train_on_batch, 10, X, y)
#endregion