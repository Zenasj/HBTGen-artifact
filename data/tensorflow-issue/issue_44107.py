import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

model.compile(optimizer=tf.keras.optimizers.Adam(lr = 0.05),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

hist = model.fit(X_train,
                 y_train,
                 batch_size = BATCH_SIZE,
                 epochs=EPOCH,
                 validation_data=(X_val, y_val),
                 shuffle = True,
                 callbacks = [ModelCheckpoint("model.hdf5",
                              monitor = 'val_loss',
                              save_best_only = True,
                              save_weights_only = False,
                              save_freq= 1,
                              verbose = 0)],
                 verbose = 0)

for key in hist.history:
  print(key)

# init some fast hand data
import numpy as np
X_train = np.random.randn(5,200,200,3)
y_train = np.array([1,2,3,4,5])
X_val = np.random.randn(5,200,200,3)
y_val = np.array([1,2,3,4,5])
#-------------------------------------------------------
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint

# init a normal model efn+dense+dense
efn = tf.keras.applications.EfficientNetB2(weights='imagenet', include_top = False)
input = Input(shape= (200,200,3))
x = efn(input)
x = Flatten()(x)
x = Dense(64, activation='relu')(x)
x = BatchNormalization()(x)
output = Dense(30, activation='softmax')(x) 
model = Model(input,output)

# compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(lr = 0.05),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
#-------------------------------------------------------
MONITOR = "val_loss" #try MONITOR = "val_sparse_categorical_accuracy" as well

hist = model.fit(X_train,
                 y_train,
                 batch_size = 64,
                 epochs=5,
                 validation_data=(X_val, y_val),
                 shuffle = True,
                 callbacks = [ModelCheckpoint("model.hdf5",
                              monitor = MONITOR,
                              save_best_only = True,
                              save_weights_only = False,
                              save_freq= 1,
                              verbose = 0)],
                 verbose = 0)
print(f"{MONITOR} in keys: {MONITOR in hist.history.keys()}", )