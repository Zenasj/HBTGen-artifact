from tensorflow.keras import layers

import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np

train = pd.read_csv("./train_v1.csv")
x_train = train[["year", 'month', 'day', 'hour', 'label']]
y_train = train['label']
test = pd.read_csv("./test_v1.csv")

def train_dataset(x):
    for i in range(0, len(x)):
        if i < 23:
            feature = x[["year", 'month', 'day', 'hour', 'label']][0:i + 1]
        else:
            feature = x[["year", 'month', 'day', 'hour', 'label']][i - 23:i + 1]
        feature = np.asarray(feature)
        feature[i, -1] = 0
        yield tf.expand_dims(feature, 0), x["label"][i]

def test_dataset(x):
    for i in range(0, len(x)):
        if i < 23:
            yield tf.expand_dims(x[["year", 'month', 'day', 'hour', 'label']][0:i], 0), 0
        else:
            yield tf.expand_dims(x[["year", 'month', 'day', 'hour', 'label']][1 - 24:i], 0), 0

model = keras.Sequential()
model.add(keras.layers.LSTM(200, input_shape=[None, 5]))
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(100))
model.add(keras.layers.Dense(1))

train_generator = train_dataset(train)
test_generator = test_dataset(test)
model.compile(loss=keras.losses.mean_absolute_error, metrics=keras.metrics.mean_absolute_error)

model.fit(train_generator, epochs=100)

pre = model.predict(test_generator)