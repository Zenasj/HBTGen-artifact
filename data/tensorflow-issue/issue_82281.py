from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
STRATEGY = tf.distribute.MirroredStrategy()
with STRATEGY.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation="gelu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128),
    ])

### Relevant log output