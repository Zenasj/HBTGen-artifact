from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import tensorflow as tf
from tensorflow import keras

print(tf.version.VERSION)

training_samples = 100
input_shape = (16, 512, 1)

dataset = tf.data.Dataset.from_tensor_slices((tf.random_uniform([32, 16, 512, 1], dtype=tf.float32), tf.random_uniform([32], dtype=tf.float32)))
dataset = dataset.shuffle(32).repeat()

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    initializer = 'he_uniform'
    nb_filts = [8, 16, 32, 400]
    out_size = 1
    model = tf.keras.models.Sequential()
    model.add(keras.layers.Conv2D(nb_filts[0], kernel_size=(3, 3),
                activation='relu', padding='same',
                kernel_initializer=initializer,
                bias_initializer=initializer, input_shape=input_shape))
    model.add(keras.layers.Conv2D(nb_filts[0], kernel_size=(3, 3),
                activation='relu', padding='same',
                kernel_initializer=initializer,
                bias_initializer=initializer))
    model.add(keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), 
                padding='same'))
    model.add(keras.layers.Conv2D(nb_filts[1], kernel_size=(3, 3),
                activation='relu', padding='same',
                kernel_initializer=initializer,
                bias_initializer=initializer))
    model.add(keras.layers.Conv2D(nb_filts[1], kernel_size=(3, 3),
                activation='relu', padding='same',
                kernel_initializer=initializer,
                bias_initializer=initializer))
    model.add(keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), 
                padding='same'))
    model.add(keras.layers.Conv2D(nb_filts[2], kernel_size=(3, 3),
                activation='relu', padding='same',
                kernel_initializer=initializer,
                bias_initializer=initializer))
    model.add(keras.layers.Conv2D(nb_filts[2], kernel_size=(3, 3),
                activation='relu', padding='same',
                kernel_initializer=initializer,
                bias_initializer=initializer))
    model.add(keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), 
                padding='same'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1024, activation='relu',
                kernel_initializer=initializer,
                bias_initializer=initializer))
    model.add(keras.layers.Dense(nb_filts[3], activation='relu',
                kernel_initializer=initializer,
                bias_initializer=initializer))
    model.add(keras.layers.Dense(out_size))

    optimizer = tf.keras.optimizers.Adam(1e-3)
    model.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=['mean_squared_error', 'mean_absolute_error'])

with strategy.scope():
    batch_size = 32
    nb_epochs = 1
    history = model.fit(dataset.batch(batch_size, drop_remainder=True), epochs=nb_epochs, steps_per_epoch=training_samples // batch_size, verbose=1)