from tensorflow.keras import layers
from tensorflow.keras import models

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
print(1)
model.add(Dense(512, activation='relu', input_shape=((28 * 28),))) #crashes here
print(2) #not printed