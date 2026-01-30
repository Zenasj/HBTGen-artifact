import random
from tensorflow.keras import layers
from tensorflow.keras import models

from tensorflow.keras.layers import Conv2D, MaxPooling2D, ReLU, Flatten, Dense, Input
from tensorflow.keras.models import Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


def cnn_model(rows, cols, channels):
    model = Model()
    inputs = Input(shape=(rows, cols, channels), dtype='float32')

    conv1 = Conv2D(64, (3, 3),activation='linear',kernel_initializer='he_uniform')(inputs)
    relu1 = ReLU()(conv1)
    pooling1 = MaxPooling2D(pool_size=(5, 5))(relu1)

    flatten = Flatten()(pooling1)

    dense1 = Dense(512)(flatten)

    relu2 = ReLU()(dense1)
    predictions = Dense(4, activation='softmax')(relu2)

    model = Model(inputs=inputs, outputs=predictions)

    return model


batch_size = 16
rows = 128
cols = 128
channels = 1
model_input = np.random.randint(0, 255, (batch_size, rows, cols, channels))
model_labels = np.random.randint(0, 1, (batch_size, 4))

model = cnn_model(rows, cols,channels)

adam = tf.train.AdamOptimizer(learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8)

model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['categorical_accuracy'])
model.fit(x=model_input,
          y=model_labels,
          epochs=2)

model.save("/home/svdvoort/test_model.hdf5")
load_model("/home/svdvoort/test_model.hdf5")

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Activation
from tensorflow.keras.models import Model

def cnn_model(rows, cols, channels):
    model = Model()
    inputs = Input(shape=(rows, cols, channels), dtype='float32')

    conv1 = Conv2D(64, (3, 3),activation='linear',kernel_initializer='he_uniform')(inputs)
    relu1 = Activation('relu')(conv1)
    pooling1 = MaxPooling2D(pool_size=(5, 5))(relu1)

    flatten = Flatten()(pooling1)

    dense1 = Dense(512)(flatten)

    relu2 = Activation('relu')(dense1)
    flatten = Flatten()(conv1)
    predictions = Dense(4, activation='softmax')(flatten)

    model = Model(inputs=inputs, outputs=predictions)

    return model