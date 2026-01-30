from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'))
model.add(layers.Dropout(rate=1.0))
model.add(layers.Flatten())
model.add(layers.Dense(units=1, activation=None))

from tensorflow.keras import layers
from tensorflow.keras import models

model = models.Sequential()
model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'))
model.add(layers.Dropout(rate=1.0))
model.add(layers.Flatten())
model.add(layers.Dense(units=1, activation=None))