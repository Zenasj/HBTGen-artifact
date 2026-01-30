from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

from gevent import monkey
monkey.patch_all()

import numpy as np
import tensorflow as tf

classifier = tf.keras.models.load_model('tensorflow_model_dir')
classifier.predict(np.array(
    np.zeros((1, 12623))
))

from gevent import monkey
monkey.patch_all()

import numpy as np
import tensorflow as tf

classifier = tf.keras.models.load_model('/opt/test/model_dir')
classifier.predict(np.array(
    np.zeros((1, 12623))
))

from gevent import monkey
monkey.patch_all()
import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test, y_test, verbose=2)