import random
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, ConvLSTM2D, MaxPooling2D, Masking, Input

inputs = Input(shape=(None,11,11,1)) 
mask = Masking(mask_value=0)(inputs)
conv2d_1 = ConvLSTM2D(64, (3, 3), activation="relu", padding="same")(mask)
max_pool_1 = MaxPooling2D((2,2), padding="same")(conv2d_1)
flatten = Flatten()(max_pool_1)
out = Dense(1, activation="sigmoid")(flatten)

x = np.random.rand(64, 300, 11, 11, 1)
y = np.random.randint(2, size=(64,1))

model = Model(inputs=inputs, outputs=out)
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.summary()
model.fit(x, y, epochs=2, batch_size=16)