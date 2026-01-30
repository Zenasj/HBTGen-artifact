import random
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

import numpy as np

X = np.random.rand(10, 4)
y = np.random.rand(10)

model = Sequential()
model.add(Dense(3, input_dim=4))
model.add(Dense(1))
model.compile(loss="mean_squared_error", optimizer=SGD(learning_rate=0.001))
model.fit(X, y)   # raises TypeError!

### Relevant log output