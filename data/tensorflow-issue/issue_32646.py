from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import Adam

model = Sequential([
    Dense(24, input_dim=4, activation='relu'),
    Dense(24, activation='relu'),
    Dense(2, activation='linear')
])
model.compile(loss='mse', optimizer=Adam(lr=0.001))

x = np.array([[-0.08623559, -0.79897248,  0.03606475,  1.09068178]])
y = np.array([[ 1.0449973,  -0.14471795]])
model.fit(x, y)

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

model = Sequential([
    Dense(24, input_dim=4, activation='relu'),
    Dense(24, activation='relu'),
    Dense(2, activation='linear')
])
model.compile(loss='mse', optimizer=Adam(lr=0.001))

x = np.array([[-0.08623559, -0.79897248,  0.03606475,  1.09068178]])
y = np.array([[ 1.0449973,  -0.14471795]])
model.fit(x, y)