from tensorflow.keras import layers

optimizer = tf.train.SomeOptimizer(learning_rate)
session.run(tf.variables_initializer(optimizer.variables()))

import os
import psutil
from tensorflow import keras
model = keras.Sequential()
model.add(keras.layers.Dense(1, input_shape=(1,)))
process = psutil.Process(os.getpid())
for count in range(1000):
    model.compile(optimizer='sgd', loss='mse')
    if count % 100 == 0:
        print('#{} : mem = {} Byte'.format(count, process.memory_info().rss))

#0 : mem = 284536832 Byte
#100 : mem = 419696640 Byte
#200 : mem = 554618880 Byte
#300 : mem = 690429952 Byte
#400 : mem = 826294272 Byte
#500 : mem = 962461696 Byte
#600 : mem = 1097363456 Byte
#700 : mem = 1234108416 Byte
#800 : mem = 1369239552 Byte
#900 : mem = 1504563200 Byte

import os
import psutil
import objgraph
from tensorflow import keras
import tensorflow as tf
print('Tensorflow: ',tf.__version__)
model = keras.Sequential()
model.add(keras.layers.Dense(1, input_shape=(1,)))
process = psutil.Process(os.getpid())
for count in range(1000):
    model.compile(optimizer='sgd', loss='mse')
    if count % 100 == 0:
        print('#{} : mem = {} Byte'.format(count, process.memory_info().rss))
        objgraph.show_growth(limit=3)