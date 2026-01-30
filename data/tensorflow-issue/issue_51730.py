from tensorflow.keras import layers
from tensorflow.keras import models

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense,  Activation
from keras.utils import np_utils

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

model = Sequential()
model.add(Dense(64, activation='relu', input_dim=25))
model.add(Dense(10, activation='softmax'))