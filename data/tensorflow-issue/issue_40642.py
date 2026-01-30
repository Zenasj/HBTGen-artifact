import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import tensorflow as tf
import os
import numpy as np

n=10
x = np.random.random((n,1))
y = np.random.random((n,1))

# set `new_cwdir` to a really long path for the two commented out lines below.
# os.mkdir(new_cwdir)
# os.chdir(new_cwdir)

all_callbacks=[]
checkpoint_filepath = "issues_with_mixed_slashes\\too_damn_buggy\\checkpoint_{epoch}"
all_callbacks.append(
    tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath, monitor='loss', verbose=1))

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(1, input_shape=x.shape[-1:]))
model.add(tf.keras.layers.Dense(1))

epochs = 2
model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanAbsoluteError())
history = model.fit(x=x, y=y, epochs=epochs, callbacks=all_callbacks)