import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import numpy as np
import tensorflow as tf

tf.keras.utils.set_random_seed(1)

# define the data
x = np.random.random(1000)
y1 = np.random.randint(0, 2, 1000)
y2 = np.random.randint(0, 2, 1000)

# define the model
input = tf.keras.layers.Input(shape=(1,))
model = tf.keras.layers.Dense(64, activation='relu')(input)
model = tf.keras.layers.Dense(2, activation='sigmoid')(model)
out1 = tf.keras.layers.Lambda(lambda x: x[..., 0], name='out1')(model)
out2 = tf.keras.layers.Lambda(lambda x: x[..., 1], name='out2')(model)
model = tf.keras.models.Model(input, [out1, out2])

# configure losses and metrics for each output
losses = {'out1': 'binary_crossentropy', 'out2': 'binary_crossentropy'}
weights = {'out1': 1.0, 'out2': 1.0}
metrics = {'out1': 'accuracy', 'out2': 'accuracy'}
model.compile(loss=losses, loss_weights=weights, optimizer='adam', metrics=metrics)

# fit
model.fit(x, [y1, y2], batch_size=10, epochs=10, verbose=2, validation_split=0.2)

import numpy as np
import tensorflow as tf

tf.keras.utils.set_random_seed(45)

# define the data
x = np.random.random(1000)
y1 = np.random.randint(0, 2, 1000)
y2 = np.random.randint(0, 2, 1000)

# define the models
def model_lambda():
    input = tf.keras.layers.Input(shape=(1,))
    model = tf.keras.layers.Dense(64, activation='relu')(input)
    model = tf.keras.layers.Dense(2, activation='sigmoid')(model)
    out1 = tf.keras.layers.Lambda(lambda x: x[..., 0], name='out1')(model)
    out2 = tf.keras.layers.Lambda(lambda x: x[..., 1], name='out2')(model)
    model = tf.keras.models.Model(input, [out1, out2])
    return model

def model_dense():
    input = tf.keras.layers.Input(shape=(1,))
    model = tf.keras.layers.Dense(64, activation='relu')(input)
    out1 = tf.keras.layers.Dense(1, activation='sigmoid', name='out1')(model)
    out2 = tf.keras.layers.Dense(1, activation='sigmoid', name='out2')(model)
    model = tf.keras.models.Model(input, [out1, out2])
    return model

for model_func, model_name in zip([model_lambda, model_dense], ['lambda model', 'dense model']):
    # configure losses and metrics for each output
    losses = {'out1': 'binary_crossentropy', 'out2': 'binary_crossentropy'}
    weights = {'out1': 1.0, 'out2': 1.0}
    metrics = {'out1': 'accuracy', 'out2': 'accuracy'}
    model = model_func()
    model.compile(loss=losses, loss_weights=weights, optimizer='adam', metrics=metrics)

    # fit
    history = model.fit(x, [y1, y2], batch_size=10, epochs=20, verbose=0, validation_split=0.2)

    # analyse
    print('----------', model_name, '----------')
    out1, out2 = model.predict(x)
    print('out1 shape:', out1.shape, ', out2 shape:', out2.shape)
    for metric in ['out1_accuracy', 'out2_accuracy', 'val_out1_accuracy', 'val_out2_accuracy']:
        accuracies = np.array(history.history[metric])
        diffs = np.diff(np.sort(accuracies))
        min_diff = np.min(diffs[diffs != 0])
        print(metric, 'is out of', np.round(1 / min_diff))

def model_lambda():
    input = tf.keras.layers.Input(shape=(1,))
    model = tf.keras.layers.Dense(64, activation='relu')(input)
    model = tf.keras.layers.Dense(2, activation='sigmoid')(model)
    out1 = tf.keras.layers.Lambda(lambda x: x[..., 0])(model)
    out1 = tf.keras.layers.Reshape((1,), name='out1')(out1)
    out2 = tf.keras.layers.Lambda(lambda x: x[..., 1])(model)
    out2 = tf.keras.layers.Reshape((1,), name='out2')(out2)
    model = tf.keras.models.Model(input, [out1, out2])
    return model