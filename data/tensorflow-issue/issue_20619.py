import random
from tensorflow import keras

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.layers import Dense
from tensorflow.python.training.adam import AdamOptimizer
import numpy as np

model = Sequential()
model.add(Dense(8, input_shape=(2, )))
model.add(Dense(1, activation='softmax'))
model.compile(optimizer=AdamOptimizer(), loss='mse')

lr_schedule = tf.keras.callbacks.ReduceLROnPlateau()

x = np.random.uniform(0, 1, (100, 2))
y = np.random.uniform(0, 1, (100, 1))
model.fit(x=x, y=y, callbacks=[lr_schedule], validation_split=0.2)

def on_epoch_begin(self, epoch, logs=None):
    # if not hasattr(self.model.optimizer, 'lr'):   <=== Original  self.model.optimizer.optimizer._learning_rate
    if not hasattr(self.model.optimizer.optimizer, '_learning_rate'):
      raise ValueError('Optimizer must have a "lr" attribute.')
    try:  # new API
      # lr = float(K.get_value(self.model.optimizer.lr))
      lr = float(self.model.optimizer.optimizer._learning_rate)
      lr = self.schedule(epoch, lr)
    except TypeError:  # Support for old API for backward compatibility
      lr = self.schedule(epoch)
    if not isinstance(lr, (float, np.float32, np.float64)):
      raise ValueError('The output of the "schedule" function '
                       'should be float.')
    # K.set_value(self.model.optimizer.lr, lr)
    self.model.optimizer.optimizer._learning_rate = lr
    if self.verbose > 0:
      print('\nEpoch %05d: LearningRateScheduler reducing learning '
            'rate to %s.' % (epoch + 1, lr))

learning_rate = K.variable(0.001)
adamW = tf.contrib.opt.AdamWOptimizer(weight_decay=1e-4,
                                      learning_rate=learning_rate, 
                                      beta1=0.9, beta2=0.999, 
                                      epsilon=1e-08, name='AdamW')
opt= TFOptimizer(adamW)
opt.lr = learning_rate
model.compile(optimizer=opt, loss=ssd_loss.compute_loss)

# Build callbacks
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='val_acc', patience=2, verbose=1, factor=0.5, min_lr=0.00001)
earlystop = EarlyStopping(patience=5)
callbacks = [reduce_lr, earlystop]

# model
from tensorflow.keras import models
from tensorflow.keras import layers 
from tensorflow.keras import optimizers

model = models.Sequential()
model.add(Dense(1024, activation='relu', input_dim=tr_x.shape[1]))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(num_classes, activation='sigmoid'))

model.compile(loss=tf.keras.losses.binary_crossentropy,
             optimizer=tf.train.AdamOptimizer(),
             metrics=['accuracy'])

history = model.fit(x=tr_x, y=tr_y, 
                    batch_size=batch_size, 
                    epochs=30, 
                    callbacks=callbacks,
                    validation_data=(val_x, val_y))

model = models.Sequential()
model.add(Dense(1024, activation='relu', input_dim=tr_x.shape[1]))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(num_classes, activation='sigmoid'))

opt = tf.keras.optimizers.Adam(lr=0.001)
model.compile(loss=tf.keras.losses.binary_crossentropy,
             optimizer=opt,
             metrics=['accuracy'])