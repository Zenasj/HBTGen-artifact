import random
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Nadam
import numpy as np

ipt = Input(shape=(4,))
out = Dense(1, activation='sigmoid')(ipt)

model = Model(ipt, out)
model.compile(optimizer=Nadam(lr=1e-4), loss='binary_crossentropy')

X = np.random.randn(32,4)
Y = np.random.randint(0,2,(32,1))
model.train_on_batch(X,Y)