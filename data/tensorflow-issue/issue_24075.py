import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

mnist_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=35,kernel_size=(3,3), strides=(1,1), padding='same', 
                           activation='relu', input_shape = (1, 28, 28), data_format="channels_first",
                           use_bias=True, bias_initializer=tf.keras.initializers.constant(0.01), 
                           kernel_initializer='glorot_normal'),
#     tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(pool_size=(2,2), padding='same', data_format='channels_first'),
    tf.keras.layers.Conv2D(filters=36,kernel_size=(3,3), strides=(1,1), padding='same', 
                           activation='relu', data_format="channels_first", use_bias=True,
                           bias_initializer=tf.keras.initializers.constant(0.01), kernel_initializer='glorot_normal'),
#     tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(pool_size=(2,2), padding='same', data_format='channels_first'),
    tf.keras.layers.Conv2D(filters=36,kernel_size=(3,3), strides=(1,1), padding='same',
                           activation='relu', data_format="channels_first", use_bias=True,
                           bias_initializer=tf.keras.initializers.constant(0.01), kernel_initializer='glorot_normal'),
#     tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(pool_size=(2,2), padding='same', data_format='channels_first'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(576, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu')
])

(mnist_images, mnist_labels), _ = tf.keras.datasets.mnist.load_data()
dataset = tf.data.Dataset.from_tensor_slices(
  (tf.cast(mnist_images[...,tf.newaxis]/255, tf.float16),
   tf.cast(mnist_labels,tf.int8)))
dataset = dataset.shuffle(1000)
mnist_images = tf.convert_to_tensor(np.expand_dims(mnist_images, axis = 1))
mnist_model.compile(optimizer=tf.keras.optimizers.Adam(), loss="categorical_crossentropy", metrics=['accuracy'])
mnist_model.fit(mnist_images, tf.one_hot(mnist_labels, depth=10), epochs=2, steps_per_epoch=100)