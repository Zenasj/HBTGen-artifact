import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

with tf.device("CPU"):
    train_dataset = tf.data.Dataset.from_tensor_slices((tf.random.uniform((1000_000,400)), tf.random.uniform((1000_000,1)))).batch(4000)

model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(400,)),
        tf.keras.layers.Dense(400, activation='relu'),
        tf.keras.layers.Dense(400, activation='relu'),
        tf.keras.layers.Dense(400, activation='relu'),
        tf.keras.layers.Dense(400, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ]
)
model.compile(
    optimizer=tf.keras.optimizers.Adam(), 
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), 
    jit_compile=True,
)

model.fit(train_dataset, 
            epochs=50, 
            verbose=True,
)

import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

with tf.device("CPU"):
    train_dataset = tf.data.Dataset.from_tensor_slices((tf.random.uniform((1000_000,400)), tf.random.uniform((1000_000,1)))).batch(4000)

model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(400,)),
        tf.keras.layers.Dense(400, activation='relu'),
        tf.keras.layers.Dense(400, activation='relu'),
        tf.keras.layers.Dense(400, activation='relu'),
        tf.keras.layers.Dense(400, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ]
)
model.compile(
    optimizer=tf.keras.optimizers.Adam(), 
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), 
    jit_compile=True,
)

model.fit(train_dataset, 
            epochs=50, 
            verbose=True,
)