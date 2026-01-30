from tensorflow.keras import layers

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.activations import sigmoid

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

X_train = np.array([0., 1, 2, 3, 4, 5], dtype=np.float32).reshape(-1,1)
Y_train = np.array([0,  0, 0, 1, 1, 1], dtype=np.float32).reshape(-1,1)

model = Sequential([
    Input(shape=(1,)),
    Dense(1, activation=sigmoid, name = "L1")
])
model.summary()

# Setting the weights and bias of the neuron
logistic_layer = model.get_layer('L1')
set_w = np.array([[2]])
set_b = np.array([-4.5])
logistic_layer.set_weights([set_w, set_b])

# Performance test
a1 = model.predict(X_train[0].reshape(1,1)) # this line caused the error
print(a1)
alog = sigmoid(np.dot(set_w,X_train[0].reshape(1,1)) + set_b)
print(alog)

def _is_distributed_dataset(ds):
#   return isinstance(ds, input_lib.DistributedDatasetInterface)
    return isinstance(ds, input_lib.DistributedDatasetSpec)

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Sequential
from tensorflow.keras.activations import sigmoid