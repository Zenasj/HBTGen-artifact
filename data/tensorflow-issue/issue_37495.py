import numpy as np
import math
import random
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.initializers import he_normal
import tensorflow_probability as tfp

# Data
np.random.seed(42)
data_X = np.random.random_sample((100,6))
data_Y = np.random.random_sample((100,1))

# Feed-forward NN
model = Sequential()

# Input Layer
model.add(Dense(6, activation='linear', input_shape=(6,)))

# Hidden Layers: Arbitrarily wide
model.add(Dense(5000, 
                activation='relu', 
                kernel_initializer=he_normal(), 
                kernel_regularizer=regularizers.l1(10**-3)
                ))
model.add(Dropout(0.3))

# Output Layer; Regression
model.add(Dense(1, activation='sigmoid'))

# Custom Loss Function
def MaxCorrelation(y_true,y_pred):
    """
    Goal is to maximize correlation between y_pred, y_true. Same as minimizing the negative.
    """
    return -tf.math.abs(tfp.stats.correlation(y_pred,y_true, sample_axis=None, event_axis=None))

# Compilation
model.compile(
            optimizer='adam', 
            loss= MaxCorrelation,
)

# Train the model
history = model.fit(data_X, data_Y,
          epochs=5,
          batch_size = 32,
          verbose=1,
         )