from tensorflow.keras import layers
from tensorflow.keras import models

3
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Sequential
model = Sequential()
model.add(Dense(32, input_shape=(16,)))
model.add(Dense(32))