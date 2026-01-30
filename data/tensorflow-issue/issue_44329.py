from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf 
import numpy as np

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(2048 * 2048, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=1, batch_size=100)