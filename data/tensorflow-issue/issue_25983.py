from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf
tf.enable_eager_execution()

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(50, input_shape=(10, )))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Dense(3))
model.add(tf.keras.layers.Activation('sigmoid'))

sgd = tf.keras.optimizers.SGD(lr=0.1, decay=0.000225, momentum=0.5)

model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])