import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D

print("Tensorflow version: {}".format(tf.__version__),flush=True)

K = 5000 # Number of images
N = 512  # Image size

MAX_SIGNAL = 5000 # The values of the training data range from 0 to this

def build_model():
  '''Create a simple test model.'''
  
  inputs = Input((N,N,1))
  s = Lambda(lambda x: x / MAX_SIGNAL) (inputs)
  s = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(s)
  outputs = s

  return Model(inputs=[inputs], outputs=[outputs])

# Generate some random data
x_train = np.random.randint(MAX_SIGNAL+1,size=(K,N,N,1),dtype=np.uint16) # Should be 2 560 000 kB
y_train = np.random.randint(1+1         ,size=(K,N,N,1),dtype=np.bool)   # Should be 1 280 000 kB
x_val   = np.random.randint(MAX_SIGNAL+1,size=(K,N,N,1),dtype=np.uint16) # Should be 2 560 000 kB
y_val   = np.random.randint(1+1         ,size=(K,N,N,1),dtype=np.bool)   # Should be 1 280 000 kB
# In total, the above arrays should be 7 680 000 kB

model = build_model()

optimizer = tf.keras.optimizers.Adam()
loss = tf.keras.losses.BinaryCrossentropy()

model.compile(optimizer=optimizer, loss=loss)
model.fit(x=x_train, y=y_train, validation_data=(x_val,y_val), batch_size=8, epochs=10)

ds_train = tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(8)
ds_val = tf.data.Dataset.from_tensor_slices((x_val,y_val)).batch(8)
model.fit(ds_train, validation_data=ds_val, epochs=10)

import tensorflow as tf
import numpy as np
import psutil
import os

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D
from tensorflow.keras.callbacks import Callback

print("Tensorflow version: {}".format(tf.__version__),flush=True)

K = 5000 # Number of images
N = 512  # Image size

MAX_SIGNAL = 5000 # The values of the training data range from 0 to this

class MemoryUsageCallback(Callback):
  '''Monitor memory usage on epoch begin and end.'''

  def on_epoch_begin(self,epoch,logs=None):
    print('**Epoch {}**'.format(epoch))
    print('Memory usage on epoch begin: {}'.format(psutil.Process(os.getpid()).memory_info().rss))

  def on_epoch_end(self,epoch,logs=None):
    print('Memory usage on epoch end:   {}'.format(psutil.Process(os.getpid()).memory_info().rss))
    
def build_model():
  '''Create a simple test model.'''
  
  inputs = Input((N,N,1))
  s = Lambda(lambda x: x / MAX_SIGNAL) (inputs)
  s = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(s)
  outputs = s

  return Model(inputs=[inputs], outputs=[outputs])

# Generate some random data
x_train = np.random.randint(MAX_SIGNAL+1,size=(K,N,N,1),dtype=np.uint16) # Should be 2 560 000 kB
y_train = np.random.randint(1+1         ,size=(K,N,N,1),dtype=np.bool)   # Should be 1 280 000 kB
x_val   = np.random.randint(MAX_SIGNAL+1,size=(K,N,N,1),dtype=np.uint16) # Should be 2 560 000 kB
y_val   = np.random.randint(1+1         ,size=(K,N,N,1),dtype=np.bool)   # Should be 1 280 000 kB
# In total, the above arrays should be 7 680 000 kB

model = build_model()

callbacks = [MemoryUsageCallback()]
optimizer = tf.keras.optimizers.Adam()
loss = tf.keras.losses.BinaryCrossentropy()

model.compile(optimizer=optimizer, loss=loss)
model.fit(x=x_train, y=y_train, validation_data=(x_val,y_val), batch_size=8, epochs=10, callbacks=callbacks, verbose=0)

ds_train = tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(8)
ds_val = tf.data.Dataset.from_tensor_slices((x_val,y_val)).batch(8)
model.fit(ds_train, validation_data=ds_val, batch_size=8, epochs=10, callbacks=callbacks, verbose=0)

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
      predictions = model(images, training=True) #Keras model subclass
      loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)