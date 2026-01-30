import random
from tensorflow.keras import layers
from tensorflow.keras import models

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

tf.debugging.set_log_device_placement(True)
X_train = np.random.rand(1000,7)
input_dim = X_train.shape[1]
epochs = [5, 10, 15]
batch_size = [10, 20, 30, 40]
param_grid = dict(epochs=epochs, batch_size=batch_size)

def create_sequential_model(activation='relu', dropout_rate=0.2, optimizer='SGD'):
    model = Sequential()
    model.add(Dense(4, input_dim=input_dim, activation=activation))
    model.add(Dropout(dropout_rate))
    model.add(Dense(2, activation=activation))
    model.add(Dense(4, activation=activation))
    model.add(Dense(7, activation='sigmoid'))
    model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
    return model

with tf.device('cpu:0'): # This works and runs on CPU
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b) # This works and runs on GPU
print(c)

with tf.device('cpu:0'): # This doesn't work
    model = KerasClassifier(build_fn=create_sequential_model, verbose=1) 
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=5)
    grid_result = grid.fit(X_train, X_train)