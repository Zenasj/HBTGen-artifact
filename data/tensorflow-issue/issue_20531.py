import random
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.callbacks import TensorBoard
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
session = tf.Session(config=config)
path_checkpoint = 'test.keras'
SEQUENCES = 5
TIME_STEPS = 10

model = Sequential()
model.add(Embedding(100, 4))
model.add(LSTM(32))
model.add(Dense(1))
model.compile(optimizer='rmsprop', loss='mse')

tensor_board = TensorBoard(log_dir='log', batch_size=2, write_graph=False,
write_grads=True, histogram_freq=4)
callback_early_stopping = EarlyStopping(monitor='val_loss',
patience=2, verbose=2)
callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
monitor='val_loss',
verbose=2,
save_weights_only=True,
save_best_only=True)
callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
factor=0.1,
min_lr=1e-4,
patience=0,
verbose=2)
x_train = np.random.randint(100, size=(SEQUENCES, TIME_STEPS), dtype='int8')

y_train = np.random.rand(SEQUENCES)

model.fit(x_train, y_train, batch_size=2, epochs=4, shuffle=True, callbacks=[callback_reduce_lr,tensor_board,callback_early_stopping,callback_checkpoint])

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.callbacks import TensorBoard
import numpy as np
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
session = tf.Session(config=config)
path_checkpoint = 'test.keras'
SEQUENCES = 5
TIME_STEPS = 10

model = Sequential()
model.add(Embedding(100, 4))
model.add(LSTM(32))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')


callback_reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                       factor=0.1,
                                       min_lr=1e-4,
                                       patience=0,
                                       verbose=2)
x_train = np.random.randint(100, size=(SEQUENCES, TIME_STEPS), dtype='int8')

y_train = np.random.rand(SEQUENCES)

model.fit(x_train, y_train, batch_size=2, epochs=4, shuffle=True, callbacks=[callback_reduce_lr])

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau