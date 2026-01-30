from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

# minimum keras example
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import TerminateOnNaN,EarlyStopping,ReduceLROnPlateau

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(8,input_shape=(16,)))
model.add(tf.keras.layers.Dense(1))
callbacks=[#TerminateOnNaN(),EarlyStopping(monitor='val_loss'),
          ReduceLROnPlateau(monitor='val_loss')]
optimizer=SGD(learning_rate=ExponentialDecay(initial_learning_rate=1.e-3,
                                            decay_steps=2,decay_rate=0.5))
model.compile(optimizer=optimizer, loss='mse')
x=tf.ones((32,16))
y=tf.ones((32,1))
v=tf.ones((8,1))
model.fit(x, y, batch_size=4, epochs=8, callbacks=callbacks,
         validation_split=0.2,steps_per_epoch=4,verbose=1)