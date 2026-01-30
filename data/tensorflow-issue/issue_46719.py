import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf
from tensorflow.python.keras.premade.linear import LinearModel
from tensorflow.python.keras.premade.wide_deep import WideDeepModel
import numpy as np

linear_model = LinearModel()
dnn_model = tf.keras.Sequential([tf.keras.layers.Dense(units=64),
                              tf.keras.layers.Dense(units=1)])
combined_model = WideDeepModel(linear_model, dnn_model)
combined_model.compile(optimizer=['sgd', 'adam'], loss='mse', metrics=['mse'])
# define dnn_inputs and linear_inputs as separate numpy arrays or
# a single numpy array if dnn_inputs is same as linear_inputs.
linear_inputs = np.random.random((2, 3))
dnn_inputs = np.random.random((2, 3))
y = np.random.randint(0, 2, (2, 2))

combined_model.fit([linear_inputs, dnn_inputs], y, epochs=1)
combined_model.save('tf-wide-deep')

import tensorflow as tf
#from tensorflow.keras.premade.linear import LinearModel
#from tensorflow.keras.premade.wide_deep import WideDeepModel
from tensorflow.keras.experimental import LinearModel
from tensorflow.keras.experimental import WideDeepModel
import numpy as np
import tensorflow_addons as tfa

linear_model = LinearModel()
dnn_model = tf.keras.Sequential([tf.keras.layers.Dense(units=64),
                             tf.keras.layers.Dense(units=1)])
combined_model = WideDeepModel(linear_model, dnn_model)
#combined_model.compile(optimizer=['sgd', 'adam'], 'mse', ['mse'])
optimizers = [
    tf.keras.optimizers.SGD(learning_rate=1e-4),
    tf.keras.optimizers.Adam(learning_rate=1e-2)
]
optimizers_and_layers = [(optimizers[0], dnn_model.layers[0:]), (optimizers[1], dnn_model.layers[1:])]
optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)
combined_model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])

# define dnn_inputs and linear_inputs as separate numpy arrays or
# a single numpy array if dnn_inputs is same as linear_inputs.

linear_inputs = np.random.random((2, 3))
dnn_inputs = np.random.random((2, 3))
y = np.random.randint(0, 2, (2, 2))

combined_model.fit([linear_inputs, dnn_inputs], y, epochs=1)

combined_model.save('tf-wide-deep')