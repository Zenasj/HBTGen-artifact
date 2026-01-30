from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

# INITIALIZE

import tensorflow as tf
import numpy as np
# define dataset
dataset = np.array([[6,148,72,35,0,33.6,0.627,50,1],
[1,85,66,29,0,26.6,0.351,31,0],
[8,183,64,0,0,23.3,0.672,32,1],
[1,89,66,23,94,28.1,0.167,21,0],
[0,137,40,35,168,43.1,2.288,33,1]])
# split into input (X) and output (y) variables
X = dataset[:,0:8]
y = dataset[:,8]

# WORKS: Model output is in range [0, 1]
# define the keras model
model = tf.keras.models.Sequential([tf.keras.layers.Dense(12, input_dim=8, activation='relu'),
                                    tf.keras.layers.Dense(1, activation='sigmoid')])
# compile the keras model
model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
              optimizer=tf.keras.optimizers.RMSprop(),
              metrics=[tf.keras.metrics.Accuracy(name='accuracy'),
                       tf.keras.metrics.Precision(name='precision'),
                       tf.keras.metrics.TruePositives(name='tp')])
# fit the keras model on the dataset
model.fit(X, y, epochs=10, batch_size=10, verbose=0)