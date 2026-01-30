import random
from tensorflow.keras import optimizers

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np

inputs = keras.Input(shape=(128,), name='digits')
x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
x = layers.Dense(64, activation='relu', name='dense_2')(x)
outputs = layers.Dense(10, activation='softmax', name='predictions')(x)
tf.print("outputs",outputs)
model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer=keras.optimizers.RMSprop(),loss=keras.losses.SparseCategoricalCrossentropy())

x_train = np.random.random_sample((100,128))
y_train = np.random.randint(0,10, (100,))
model.fit(x_train, y_train,epochs=10)

inputs = keras.Input(shape=(128,), name='digits')
x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
x = layers.Dense(64, activation='relu', name='dense_2')(x)
outputs = layers.Dense(10, activation='softmax', name='predictions')(x)
tf.print("outputs",outputs)
model = keras.Model(inputs=inputs, outputs=outputs)