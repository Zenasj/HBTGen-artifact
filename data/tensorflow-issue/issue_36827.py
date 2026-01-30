import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

def zero_loss(y_true, y_pred):
    return 0.
model.compile(optimizer=tf.keras.optimizers.Adam(), 
              loss={'main_output': tf.keras.losses.MeanSquaredError(), 'second_output': zero_loss}, 
              metrics={'second_output': tf.keras.losses.MeanSquaredError()})

import tensorflow as tf
import numpy as np

print('Using Tensorflow version {} (git version {})'.format(tf.version.VERSION, tf.version.GIT_VERSION))

input_data = np.random.normal(size=(1024, 5))
target_data = {'main_output' : np.random.normal(size=(1024,)), 'second_output': np.random.normal(size=(1024,))}
input_dataset = tf.data.Dataset.from_tensor_slices(input_data)
target_dataset = tf.data.Dataset.from_tensor_slices(target_data)
dataset = tf.data.Dataset.zip((input_dataset, target_dataset)).batch(16)

def make_model():
    inp = tf.keras.Input(shape=(5))
    hidden = tf.keras.layers.Dense(24, activation='relu')(inp)
    out = tf.keras.layers.Dense(1, name='main_output')(hidden)
    out2 = tf.keras.layers.Lambda(lambda x: x ** 2, name='second_output')(out)
    model = tf.keras.Model(inputs=inp, outputs=[out, out2])
    return model

epochs = 2

model = make_model()
model.compile(optimizer=tf.keras.optimizers.Adam(), 
              loss={'main_output': tf.keras.losses.MeanSquaredError(),}, 
              metrics={'second_output': tf.keras.losses.MeanSquaredError()})
model.fit(x=dataset, epochs=epochs)

model = make_model()
model.compile(optimizer=tf.keras.optimizers.Adam(), 
              loss={'main_output': tf.keras.losses.MeanSquaredError(), 'second_output': None}, 
              metrics={'second_output': tf.keras.losses.MeanSquaredError()})
model.fit(x=dataset, epochs=epochs)

model = make_model()
def zero_loss(y_true, y_pred):
    return 0.
model.compile(optimizer=tf.keras.optimizers.Adam(), 
              loss={'main_output': tf.keras.losses.MeanSquaredError(), 'second_output': zero_loss}, 
              metrics={'second_output': tf.keras.losses.MeanSquaredError()})
model.fit(x=dataset, epochs=epochs)