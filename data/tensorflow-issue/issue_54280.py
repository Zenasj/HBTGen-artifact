import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import tensorflow as tf

X, y = tf.random.uniform((50,)), tf.random.uniform((50,))
X_val, y_val = tf.random.uniform((5,)), tf.random.uniform((5,))

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(schedule=lambda epoch,lr: lr*0.995, verbose=1)

model = tf.keras.models.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])

model.compile(optimizer=tf.keras.optimizers.SGD(), loss='mse')

history = model.fit(X, y, epochs=1, validation_data=(X_val,y_val), callbacks=[lr_scheduler])

model.save_weights('tmp_model')

model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=history.history['lr'][-1]*0.1),
    #optimizer=tf.keras.optimizers.SGD(learning_rate=float(history.history['lr'][-1]*0.1)), # work-around
    loss='mse'
)

model.fit(X, y, epochs=1, validation_data=(X_val,y_val), callbacks=[lr_scheduler])
model.load_weights('tmp_model')