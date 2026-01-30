import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import tensorflow as tf
import numpy as np
import time
    
xTrain = np.random.randn(2048, 128)
yTrain = np.random.randn(2048, 128)
xValid = np.random.randn(256,  128)
yValid = np.random.randn(256,  128)
                
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape = (128,)),
    tf.keras.layers.Dense(128, activation = "tanh"),
    tf.keras.layers.Dense(128)
])      
        
model.compile(
    optimizer = tf.keras.optimizers.Adam(1.0e-3),
    loss = tf.keras.losses.MeanSquaredError(),
)   
    
# Preliminary run to build graphs.
model.fit(x = xTrain, y = yTrain, epochs = 5, batch_size = 2048,
    validation_data = (xValid, yValid)
)

t1 = time.time()

model.fit(x = xTrain, y = yTrain, epochs = 100, batch_size = 2048,
    validation_data = (xValid, yValid)
)

t2 = time.time()

model.fit(x = xTrain, y = yTrain, epochs = 100, batch_size = 2048)

t3 = time.time()

print("Time with validation:     %.2f s" % (t2 - t1))
print("Time without validation:  %.2f s" % (t3 - t2))