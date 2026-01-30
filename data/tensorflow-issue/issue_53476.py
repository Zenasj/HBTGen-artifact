import random
from tensorflow.keras import optimizers

import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

inputs = keras.Input(shape=(6400,), name='digits')
num_units = 8192

dense1 = layers.Dense(num_units, activation='relu', name='dense_1')
x = dense1(inputs)
dense2 = layers.Dense(num_units, activation='relu', name='dense_2')
x = dense2(x)

x = layers.Dense(1000, name='dense_logits')(x)
outputs = layers.Activation('relu', dtype='float32', name='predictions')(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=keras.optimizers.RMSprop(),
              metrics=['accuracy'])

x_train = x_train.reshape(60000, 6400).astype('float32') / 255
y_train = np.random.randint(10, size=60000)

history = model.fit(x_train, y_train,
                    batch_size=8192,
                    epochs=5,
                    validation_split=0.2)