from tensorflow import keras
from tensorflow.keras import layers

import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['NCCL_DEBUG'] = 'WARN' 
import tensorflow as tf
import pathlib


print("###########################")
print("Python version: ", sys.version)
print("Tensorflow version: ", tf.version.GIT_VERSION, tf.version.VERSION)
print("Physical devices found: ",tf.config.list_physical_devices())
#Choose mirrored strategy
strategy = tf.distribute.MirroredStrategy()
print("Strategy found number of devices: ", strategy.num_replicas_in_sync)
print("###########################")



datadir = pathlib.Path("./data/")
# Preprocessing
train_ds = tf.keras.utils.image_dataset_from_directory(
    directory=datadir,
    batch_size=32,
    image_size=(300,300),
    seed=123,
    validation_split=0
)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = 2

with strategy.scope():
    model = tf.keras.Sequential([
        #Rescale images before feeding them to the network
        tf.keras.layers.Rescaling(1./255),
        #create a small convolutional network
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes)
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )

model.fit(
    train_ds,
    epochs=50
)