import numpy as np
import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle

data, targets = shuffle(*fetch_openml('CIFAR_10', version=1, return_X_y=True)) # same happens if I force these to be float32s
train_sz = 50000
X_train, X_test, y_train, y_test = data[:train_sz, :], data[train_sz:, :], np.asarray(targets[:train_sz], dtype=np.int), np.asarray(targets[train_sz:], dtype=np.int)

model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(X_train.shape[1],)))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(10))
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='adam')

s = 0
for _ in range(500):
    for i in range(100):
        layers = []
        for layer in model.get_weights():
            layers.append(np.random.normal(0, 1, layer.shape))
        model.set_weights(layers)
        eval = model.evaluate(X_train, y_train)
        s += eval
        print(f'Done {i}')
print(s)

preds = model(X_train)
eval = my_loss(y_train, preds)

s = 0
for j in range(100):
    for i in range(100):
        layers = []
        for layer in model.get_weights():
            layers.append(np.random.normal(0, 1, layer.shape))
        model.set_weights(layers)
        eval = model.evaluate(X_train, y_train)
        s += eval
        tf.keras.backend.clear_session()
        gc.collect()
        print(f'Done {i}')
    print('*'*100)
    print(f'Done {j}')
        
print(s)