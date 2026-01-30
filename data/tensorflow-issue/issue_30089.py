import random
from tensorflow.keras import layers
from tensorflow.keras import optimizers

from tensorflow.python.keras.utils.data_utils import Sequence
class mygenerator(Sequence):
    def __init__(self, x_set_1, x_set_2, y_set, batch_size = 16):
        self.x1, self.x2, self.y = x_set_1, x_set_2, y_set
        self.batch_size = batch_size
        self.len_data = 0

    def __len__(self):
        return int(np.ceil(len(self.y) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x1 = self.x1[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x2 = self.x2[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        # read your data here using the batch lists, batch_x and batch_y
        xi = [item for item in batch_x1] 
        xj = [item for item in batch_x2] 
        yi = [item for item in batch_y]
        return np.asarray([xi, xj], yi)

model.fit_generator(mygenerator(neural_network_x_1, neural_network_x_2, neural_network_y), steps_per_epoch= 25,epochs=100)

import numpy as np
no_element = 10
x1 = np.random.rand(no_element ,512)
x2 = np.random.rand(no_element ,512)
x3 = np.random.randint(2, size=no_element)

from tensorflow.python.keras.utils.data_utils import Sequence

class mygenerator(Sequence):
    def __init__(self, x_set_1, x_set_2, y_set, batch_size = 16):
        self.x1, self.x2, self.y = x_set_1, x_set_2, y_set
        self.batch_size = batch_size
        self.len_data = 0

    def __len__(self):
        return int(np.ceil(len(self.y) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x1 = self.x1[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x2 = self.x2[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        # read your data here using the batch lists, batch_x and batch_y
        xi = [item for item in batch_x1] 
        xj = [item for item in batch_x2] 
        yi = [item for item in batch_y]
        return np.asarray([xi, xj], yi)


from keras import Sequential
from keras import Input
from keras.layers import Dense, Concatenate
from keras import Model
# from keras import optimizers
from keras.optimizers import Adam

# define two sets of inputs
inputA = Input(shape=(512,))
inputB = Input(shape= (512,))

# the first branch operates on the first input
x = Dense(128, activation="relu")(inputA)
x = Model(inputs=inputA, outputs=x)

# the second branch opreates on the second input
y = Dense(128, activation="relu")(inputB)
y = Model(inputs=inputB, outputs=y)

# combine the output of the two branches
combined = Concatenate()([x.output, y.output])

# apply a FC layer and then a regression prediction on the
# combined outputs
z = Dense(16, activation="relu")(combined)
z = Dense(4, activation="relu")(combined)
z = Dense(2, activation="linear")(z)

model = Model(inputs=[x.input, y.input], outputs=z)
opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss="mean_absolute_percentage_error", optimizer=opt)


model.fit_generator(mygenerator(x1, x2, x3), steps_per_epoch= 25,epochs=100)