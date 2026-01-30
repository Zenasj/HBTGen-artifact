from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
import numpy as np
import pickle
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

IMG_SIZE = 50

def prepare(file):
    img_array = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

    predictdata = tf.reshape(new_array, (1, 50, 50))
    predictdata = np.expand_dims(predictdata, -1)
    return predictdata


pickle_ind = open("x.pickle", "rb")
x = pickle.load(pickle_ind)
x = np.array(x, dtype=float)
x = np.expand_dims(x, -1)

pickle_ind = open("y.pickle", "rb")
y = pickle.load(pickle_ind)

n_batch = len(x)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(1, activation='softmax'))

model.summary()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x, y, epochs=1, batch_size=n_batch)
prediction = model.predict([prepare('demo1.jpg')], batch_size=n_batch, steps=1, verbose=1)

print(prediction)