from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

import tensorflow
import numpy as np
import string
import random

from tensorflow.keras.models import Sequential
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.layers import Input, Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam

training_size = 12528566  # if you change the size to 1000 it will work if to_use_sparse is False or is_convert is False

to_use_sparse = True  # if set True and to_convert is False, throws error that adapter is not found

X_train = np.random.randint(low=0, high=169999, size=(training_size, 10), dtype='int32')
labels = []
for i in range(3811):
    labels.append(''.join(random.choices(string.ascii_uppercase + string.digits, k=6)))
Y = [random.choice(labels) for i in range(training_size)]

if to_use_sparse:
    Y_train = LabelBinarizer(sparse_output=True).fit(labels).transform(Y)  # no adapter is found
else:
    Y_train = LabelBinarizer(sparse_output=True).fit(labels).transform(Y).toarray()  # this thing works but only if training_size is small, say 1000

    
to_convert = True # converting the labels to integers of their index and using sparse_categorical_crossentropy

if to_convert and to_use_sparse: # if true, it works as expected
    model = Sequential()
    model.add(Embedding(input_dim=170000, output_dim=100, input_length=10))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(3811, 'softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=0.001))
    model.fit(X_train,
              np.asarray(Y_train.tocoo().col),
              batch_size=16384,
              epochs=5,
              verbose=1)
else:
    model = Sequential()
    model.add(Embedding(input_dim=170000, output_dim=100, input_length=10))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(3811, 'softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001))
    model.fit(X_train,
              Y_train,
              batch_size=16384,
              epochs=5,
              verbose=1)