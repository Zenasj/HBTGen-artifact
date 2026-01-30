import random
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import numpy as np
from tensorflow.keras.layers import Dense, Input, Layer
from tensorflow.keras.models import Model

tf.enable_eager_execution()

input_tensor = Input(shape=(20,), name="input")
hidden = Dense(100, activation='relu')(input_tensor)
out1 = Dense(10, activation='relu', name="out1")(hidden)
out2 = Dense(5, activation='relu', name="out2")(hidden)
model = Model(inputs=input_tensor, outputs=[out1, out2])
model.compile(loss={"out1": "mse"}, optimizer=tf.train.AdamOptimizer(learning_rate=0.001))
model.summary()

np.random.seed(0)
X = np.random.random((3, 20)).astype(np.float32)
Y1 = np.random.random((3, 10)).astype(np.float32)
Y2 = np.random.random((3, 5)).astype(np.float32)
model.fit(x={'input' : X}, y={'out1' : Y1}, batch_size=1, epochs=10)

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import numpy as np
from tensorflow.keras.layers import Dense, Input, Layer
from tensorflow.keras.models import Model

tf.enable_eager_execution()

def zero_loss(y_true, y_pred):
  return tf.constant(0.0)

input_tensor = Input(shape=(20,), name="input")
hidden = Dense(100, activation='relu')(input_tensor)
out1 = Dense(10, activation='relu', name="out1")(hidden)
out2 = Dense(5, activation='relu', name="out2")(hidden)
model = Model(inputs=input_tensor, outputs=[out1, out2])
model.compile(loss={"out1": "mse", "out2": zero_loss}, optimizer=tf.train.AdamOptimizer(learning_rate=0.001))
model.summary()

np.random.seed(0)
X = np.random.random((3, 20)).astype(np.float32)
Y1 = np.random.random((3, 10)).astype(np.float32)
Y2 = np.random.random((3, 5)).astype(np.float32)
model.fit(x={'input' : X}, y={'out1' : Y1, 'out2': Y2}, batch_size=1, epochs=10)