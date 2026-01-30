import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import pickle
import numpy as np

tf.random.set_seed(42)

input_x = np.random.randint(0, 50000, (10000,1))
input_y = np.random.randint(0, 50000, (10000,1))
output = input_x + input_y
input = np.concatenate((input_x, input_y), axis=1)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(2, activation = tf.keras.activations.relu, input_shape=[2]),   
    tf.keras.layers.Dense(2, activation = tf.keras.activations.relu),
    tf.keras.layers.Dense(1),
])

model.compile(loss = tf.keras.losses.mae,
              optimizer=tf.optimizers.Adam(learning_rate=0.00001),
              metrics = ['mse'])
              
model.fit(input, output, epochs = 100)

fl = open('D:/tf/tf.pkl', 'wb')
pickle.dump(model, fl)
fl.close()

fl = open('D:/tf/tf.pkl', 'rb')
model = pickle.load(fl)
print(model.predict([[2.2, 5.1]]))
fl.close()