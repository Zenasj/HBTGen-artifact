import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf
import datetime
import os

class SimpleModel(tf.keras.Model):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.Dense_in = tf.keras.layers.Dense(units = 50, 
                                activation=tf.nn.relu, 
                                use_bias=True)
        self.Dense_hidden = tf.keras.layers.Dense(units = 20, 
                                activation=tf.nn.relu, 
                                use_bias=True)
        self.Dense_out = tf.keras.layers.Dense(units = 1, 
                                activation=None, 
                                use_bias=True)

    def call(self, inputs, training = True):
        X = self.Dense_in(inputs)
        X = self.Dense_hidden(X)
        X = self.Dense_out(X)
        return X

dummy_data = tf.random.normal((10000, 50))
dummy_y = tf.random.uniform((10000,), minval=0, maxval=2, dtype=tf.dtypes.int32)


#### LR is not logged here ####
model = SimpleModel()

loss = tf.keras.losses.BinaryCrossentropy(name='binary_crossentropy', from_logits=True)
opt = tf.keras.optimizers.Adam(learning_rate = 0.001)
model.compile(optimizer = opt, loss = loss, metrics=["acc"])

_datetime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join('dummy_logs', _datetime)
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

model.fit(x = dummy_data, y = dummy_y, epochs = 10,
        callbacks=[tf.keras.callbacks.ReduceLROnPlateau(monitor='acc', 
                        factor=0.5, patience=10, verbose=0, mode='auto', 
                        min_delta=0.0001, cooldown=0, min_lr=0),
                    tb_callback])

#### LR is logged here ####
model = SimpleModel()

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate = 0.001,
    decay_steps=100000,
    decay_rate=0.96)

loss = tf.keras.losses.BinaryCrossentropy(name='binary_crossentropy', from_logits=True)
opt = tf.keras.optimizers.Adam(learning_rate = lr_schedule)
model.compile(optimizer = opt, loss = loss, metrics=["acc"])

_datetime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join('dummy_logs', _datetime)
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

model.fit(x = dummy_data, y = dummy_y, epochs = 10,
        callbacks=[tb_callback])

def _collect_learning_rate(self, logs):
    lr_schedule = getattr(self.model.optimizer, 'lr', None)
    if isinstance(lr_schedule, learning_rate_schedule.LearningRateSchedule):
      logs['learning_rate'] = lr_schedule(self.model.optimizer.iterations)
    return logs

class TensorBoardCallback(tf.keras.callbacks.TensorBoard):
    def __init__(self, **kwargs):
        super(TensorBoardCallback, self).__init__(**kwargs)

    def _collect_learning_rate(self, logs):
        lr_schedule = getattr(self.model.optimizer, 'lr', None)
        if isinstance(lr_schedule, tf.keras.optimizers.schedules.LearningRateSchedule):
            logs['learning_rate'] = lr_schedule(self.model.optimizer.iterations)
        elif lr_schedule is not None:
            logs['learning_rate'] = lr_schedule
        return logs