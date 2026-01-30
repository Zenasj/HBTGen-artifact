from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.utils import to_categorical

import tensorflow as tf
import numpy as np

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train, X_test = X_train.astype(float)/255, X_test.astype(float)/255
X_train, X_test = X_train.reshape(len(X_train),28,28,1), X_test.reshape(len(X_test),28,28,1)
y_train, y_test = to_categorical(y_train), to_categorical(y_test)

# model definition
model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(16, 8,
                             strides=2,
                             padding='same',
                             activation='relu',
                             input_shape=(28, 28, 1)),
      tf.keras.layers.MaxPool2D(2, 1),
      tf.keras.layers.Conv2D(32, 4,
                             strides=2,
                             padding='valid',
                             activation='relu'),
      tf.keras.layers.MaxPool2D(2, 1),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(32, activation='relu'),
      tf.keras.layers.Dense(10, activation="softmax")
  ])

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

# this causes errors
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

# using the keyword, everything works
# loss = "categorical_crossentropy" 

# Compile model with Keras
model.compile(optimizer=optimizer, loss=loss, metrics=['categorical_accuracy'])

# Train model with Keras
model.fit(X_train, y_train, epochs=5, batch_size=1000,
          validation_data=(X_test, y_test), verbose=2)