from tensorflow.keras import layers
from tensorflow.keras import models

from tensorflow.keras.layers import Dense, Embedding, Flatten, Lambda, Subtract, Input, Concatenate, Average, Reshape, GlobalAveragePooling1D, Dot, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import Sequence
from tensorflow.keras import initializers

import tensorflow_datasets as tfds
tfds.list_builders()
dataset, info = tfds.load("mnist", with_info=True)
inputs = Input((28, 28, 1), name="image")
First = Dense(128, activation="relu")
Second = Dropout(0.2)
Third = Dense(10, activation="softmax", name="label")

first = First(inputs)
second = Second(first)
third = Third(second)
model = Model(inputs=[inputs], outputs=[third])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(dataset['train'].batch(4096))

def autoencoder_sample(x):
    return x, x