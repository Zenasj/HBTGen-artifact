from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import pathlib

import tensorflow as tf

def main():
    path = "\\\\localhost\\c$"
    path = pathlib.Path(path)
    path = path/'test'

    path.mkdir(exist_ok=True, parents=True)

    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(16,)))
    model.add(tf.keras.layers.Dense(8))

    model.save(path/'model.keras')

    model = tf.keras.models.load_model(path/'model.keras')

if __name__ == "__main__":
    main()

import os
import pathlib

os.environ["TF_USE_LEGACY_KERAS"]="1"
import tensorflow as tf

def main():
    path = "\\\\localhost\\c$"
    path = pathlib.Path(path)
    path = path/'test'

    path.mkdir(exist_ok=True, parents=True)

    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(16,)))
    model.add(tf.keras.layers.Dense(8))

    model.save(path/'model.keras')

    model = tf.keras.models.load_model(path/'model.keras')

if __name__ == "__main__":
    main()