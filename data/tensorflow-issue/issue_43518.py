import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

x = np.random.rand(10000, 10)
y = np.random.choice([0, 1], (10000, ))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)


model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10)

model.save('./test_model/')

loaded_model = tf.keras.models.load_model('./test_model/')

print(model.evaluate(x_test, y_test))
print(loaded_model.evaluate(x_test, y_test))