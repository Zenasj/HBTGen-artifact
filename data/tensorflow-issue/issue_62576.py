import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

#example.py
import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense, GRU, Input, Lambda, Dropout, Layer
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.utils import Sequence


maxValue = 1 - tf.keras.backend.epsilon()

dataSize = 100
BS = 10
dataShape = (10, 2)
outShape = 1

def getModel():
    input_layer = Input(shape=dataShape, name="input")
    x = input_layer
    x = GRU(units=10, unroll=True, name="gru")(x)
    dropoutTraining = True
    x1 = Dropout(maxValue, name="head1")(x, training=dropoutTraining)
    x2 = Dropout(0, name="head2")(x, training=dropoutTraining)
    out1 = Dense(outShape, use_bias=False)(x1)
    out2 = Dense(outShape, use_bias=False)(x2)
    model = KerasModel(input_layer, [out1, out2])
    model.summary()

    model.compile(
        loss="binary_crossentropy",
        optimizer='Adam',
    )
    return model


class DataGenerator(Sequence):
    def __init__(self, model):
        self.model = model
        self.id = 0

    def __len__(self):
        return 10

    def __getitem__(self, index):
        X = np.random.rand(BS, *dataShape)
        out_vec1 = np.random.rand(BS, outShape)
        out_vec2 = np.random.rand(BS, outShape)
        return X, [out_vec1, out_vec2]

    def on_epoch_end(self):
        self.id+=1
        X = np.random.rand(1, *dataShape)

        y = self.model.predict(X)
        print("Before switch", y)

        head1 = self.model.get_layer('head1')
        head2 = self.model.get_layer('head2')
        if self.id % 2 == 0:
            head1.rate = maxValue
            head2.rate = 0
        else:
            head1.rate = 0
            head2.rate = maxValue

        y = self.model.predict(X)
        print("After switch", y)

model = getModel()

gen = DataGenerator(model)
steps_per_epoch = dataSize // BS


history = model.fit_generator(
    gen,
    steps_per_epoch = steps_per_epoch,
    epochs = 10,
    verbose = 1)


#example2.py
import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense, GRU, Input, Lambda, Dropout, Layer
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.utils import Sequence


maxValue = 1 - tf.keras.backend.epsilon()

dataSize = 100
BS = 10
dataShape = (10, 2)
outShape = 1


class ConstMul(Layer):
    def __init__(self, const_val, *args, **kwargs):
        super().__init__(*args,  **kwargs)
        self.const = const_val

    def call(self, inputs, **kwargs):
        return inputs * self.const


def getModel():
    input_layer = Input(shape=dataShape, name="input")
    x = input_layer
    x = GRU(units=10, unroll=True, name="gru")(x)
    dropoutTraining = True
    x1 = ConstMul(1, name="head1")(x, training=dropoutTraining)
    x2 = ConstMul(0, name="head2")(x, training=dropoutTraining)
    out1 = Dense(outShape, use_bias=False)(x1)
    out2 = Dense(outShape, use_bias=False)(x2)
    model = KerasModel(input_layer, [out1, out2])
    model.summary()

    model.compile(
        loss="binary_crossentropy",
        optimizer='Adam')
    return model


class DataGenerator(Sequence):
    def __init__(self, model):
        self.model = model
        self.id = 0

    def __len__(self):
        return 10

    def __getitem__(self, index):
        X = np.random.rand(BS, *dataShape)
        out_vec1 = np.random.rand(BS, outShape)
        out_vec2 = np.random.rand(BS, outShape)
        return X, [out_vec1, out_vec2]

    def on_epoch_end(self):
        self.id+=1
        X = np.random.rand(1, *dataShape)

        y = self.model.predict(X)
        print("Before switch", y)

        head1 = self.model.get_layer('head1')
        head2 = self.model.get_layer('head2')
        if self.id % 2 == 0:
            head1.const = maxValue
            head2.const = 0
        else:
            head1.const = 0
            head2.const = maxValue

        y = self.model.predict(X)
        print("After switch", y)


model = getModel()

gen = DataGenerator(model)
steps_per_epoch = dataSize // BS


history = model.fit_generator(
    gen,
    steps_per_epoch = steps_per_epoch,
    epochs = 10,
    verbose = 1)

#example3.py
import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense, GRU, Input, Lambda, Dropout, Layer
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.utils import Sequence


maxValue = 1 - tf.keras.backend.epsilon()

dataSize = 100
BS = 10
dataShape = (10, 2)
outShape = 1


class ConstMul(Layer):
    def __init__(self, const_val, *args, **kwargs):
        super().__init__(*args,  **kwargs)
        self.const = const_val

    def call(self, inputs, **kwargs):
        return inputs * self.const


def getModel():
    input_layer = Input(shape=dataShape, name="input")
    x = input_layer
    x = GRU(units=10, unroll=True, name="gru")(x)
    dropoutTraining = True
    x1 = ConstMul(1, name="head1")(x, training=dropoutTraining)
    x2 = ConstMul(0, name="head2")(x, training=dropoutTraining)
    out1 = Dense(outShape, use_bias=False)(x1)
    out2 = Dense(outShape, use_bias=False)(x2)
    model = KerasModel(input_layer, [out1, out2])
    model.summary()

    model.compile(
        loss="binary_crossentropy",
        optimizer='Adam')
    return model


class DataGenerator(Sequence):
    def __init__(self, model):
        self.model = model
        self.id = 0

    def __len__(self):
        return 10

    def __getitem__(self, index):
        X = np.random.rand(BS, *dataShape)
        out_vec1 = np.random.rand(BS, outShape)
        out_vec2 = np.random.rand(BS, outShape)
        return X, [out_vec1, out_vec2]

    def on_epoch_end(self):
        self.id+=1
        X = np.random.rand(1, *dataShape)

        y = self.model.predict(X)
        print("Before switch", y)

        head1 = self.model.get_layer('head1')
        head2 = self.model.get_layer('head2')
        if self.id % 2 == 0:
            head1.const = maxValue
            head2.const = 0
        else:
            head1.const = 0
            head2.const = maxValue

        self.model.compile(
            loss="binary_crossentropy",
            optimizer='Adam')

        y = self.model.predict(X)
        print("After switch", y)


model = getModel()

gen = DataGenerator(model)
steps_per_epoch = dataSize // BS


history = model.fit_generator(
    gen,
    steps_per_epoch = steps_per_epoch,
    epochs = 10,
    verbose = 1)