import random
from tensorflow import keras
from tensorflow.keras import layers

from memory_profiler import profile
from time import time
import numpy as np
import tensorflow as tf

model = tf.keras.Sequential([tf.keras.layers.Dense(100, activation=tf.nn.softmax)])
model.compile(loss='mse', optimizer='sgd')

@profile
def eval(x, y):
    model.evaluate(x, y)

x = np.random.normal(size=(1,100))
y = np.random.normal(size=(1,100))

for i in range(100000):
    print('iteration', i)
    tic = time()
    eval(x, y)
    print('timeit', time() - tic)

from memory_profiler import profile
from time import time
import numpy as np
import tensorflow as tf

model = tf.keras.Sequential([tf.keras.layers.Dense(100, activation=tf.nn.softmax)])
model.compile(loss='mse', optimizer='sgd')

@profile
def eval(dataset):
    model.evaluate(dataset)

x = np.random.normal(size=(1,100))
y = np.random.normal(size=(1,100))

dataset = tf.data.Dataset.from_tensor_slices((x, y))
dataset = dataset.batch(1)

for i in range(100000):
    print('iteration', i)
    tic = time()
    eval(dataset)
    print('timeit', time() - tic)