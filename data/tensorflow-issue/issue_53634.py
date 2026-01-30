import random
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import KFold
import numpy as np

input_dim = 256
input_x = Input(input_dim,)
x = Dense(128, activation='relu')(input_x)
p = Dense(2, activation='sigmoid')(x)
model = Model(inputs=input_x, outputs=p)

model.compile(loss='mse', optimizer=Adam(1e-5), metrics = ['acc'])

sample_num = 2000
train, label = np.random.random((sample_num, input_dim)), np.random.random((sample_num, 2))
kfold = KFold(n_splits=5)
for i, (train_ind, valid_ind) in enumerate(kfold.split(list(range(sample_num)))):
    model_path = f'model-{i}.h5'
    ckpt = ModelCheckpoint(model_path, monitor='val_loss')
    model.fit([train[train_ind], label[train_ind]], label[train_ind], validation_data=[train[valid_ind], label[valid_ind]], epochs=10, batch_size=64, callbacks=[ckpt])

import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import KFold
import numpy as np

input_dim = 256
input_x = Input(input_dim,)
x = Dense(128, activation='relu')(input_x)
p = Dense(2, activation='sigmoid')(x)
model = Model(inputs=input_x, outputs=p)

model.compile(loss='mse', optimizer=Adam(1e-5), metrics = ['acc'])

sample_num = 2000
train, label = np.random.random((sample_num, input_dim)), np.random.random((sample_num, 2))
kfold = KFold(n_splits=5)
for i, (train_ind, valid_ind) in enumerate(kfold.split(list(range(sample_num)))):
    model_path = f'model_{i}.h5'
    ckpt = ModelCheckpoint(model_path, monitor='val_loss')
    model.fit(train[train_ind], label[train_ind], validation_data=[train[valid_ind], label[valid_ind]], epochs=10, batch_size=64, callbacks=[ckpt])