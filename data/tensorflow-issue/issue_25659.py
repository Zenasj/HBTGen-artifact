from tensorflow.keras import layers
from tensorflow.keras import models

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv1D, Reshape, Concatenate

branch1 = Sequential()
branch1.add(Conv1D(2,3,activation='tanh',input_shape=(100, 1)))
branch1.add(Conv1D(4,3,activation='tanh'))
branch1.add(Conv1D(6,3,activation='tanh'))
branch1.add(Dropout(0.2))
branch1.add(Conv1D(8,3,activation='tanh'))
branch1.add(Conv1D(10,3,activation='relu'))
branch1.add(Flatten())
branch1.add(Dense(10))


branch2 = Sequential()
branch2.add(Dense(10,input_dim=1))
branch2.add(Dense(10,activation='linear'))

# Concatenate([branch1,branch2])


model = Sequential()
model.add(Concatenate([branch1, branch2]))
model.add(Dense(1,activation='relu'))
# branch1.add(Dropout(0.4))
model.add(Dense(1,activation='relu'))
model.compile(loss='mean_squared_error', optimizer='rmsprop')

model_info = model.fit([data_cnn,data_mw], target, epochs=1000, batch_size=100, verbose=2,validation_data=([val_data_cnn,val_data_mw],val_target))