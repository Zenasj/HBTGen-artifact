from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.saving import hdf5_format

model = Sequential()
model.add(Dense(10, input_dim=100))
model.compile(optimizer=Adam())
hdf5_format.save_model_to_hdf5(model, "test.h5")