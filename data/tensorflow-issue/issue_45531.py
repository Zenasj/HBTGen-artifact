from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np

print(tf.__version__)

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=1)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

model.save('model.h5')

model = tf.keras.models.load_model('model.h5')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

model = tf.keras.models.load_model('model.h5', compile=False)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)