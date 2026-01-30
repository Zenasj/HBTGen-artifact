import random
from tensorflow.keras import layers
from tensorflow.keras import models

from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import numpy as np

class DataGenerator(Sequence):
    def __init__(self, batch_size = 512, shape = (10,)):
        self.shape = shape
        self.batch_size = batch_size

    def on_epoch_end(self):
        # do nothing
        pass

    def __getitem__(self, idx):
        print("[+] Idx: %d" % idx)
        y = np.ones((10,2))
        y[:,0] = 0
        return np.random.random((10,10)), y

    def __len__(self):
        return 7

inp = Input(shape = (10,), dtype = np.float)
x = Dense(10, activation = 'relu')(inp)
out = Dense(2, activation='softmax')(x)
model = Model(inputs=inp, outputs= out)

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

generator = DataGenerator()

model.fit_generator(generator, epochs = 2, shuffle = False, workers = 0)

from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import numpy as np

class DataGenerator(Sequence):
    def __init__(self, batch_size = 512, shape = (10,)):
        self.shape = shape
        self.batch_size = batch_size
        self.first_epoch = True

    def on_epoch_end(self):
        # do nothing
        pass

    def __getitem__(self, idx):
        print("[+] Idx: %d" % idx)
        y = np.ones((10,2))
        y[:,0] = 0
        if self.first_epoch:
               self.first_epoch = False
               self.on_epoch_end()
        return np.random.random((10,10)), y

    def __len__(self):
        return 7

inp = Input(shape = (10,), dtype = np.float)
x = Dense(10, activation = 'relu')(inp)
out = Dense(2, activation='softmax')(x)
model = Model(inputs=inp, outputs= out)

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

generator = DataGenerator()

model.fit_generator(generator, epochs = 2, shuffle = False, workers = 0)