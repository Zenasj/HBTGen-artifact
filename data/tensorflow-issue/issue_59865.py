import tensorflow as tf
from tensorflow import keras

mirrored_strategy = tf.distribute.MirroredStrategy()

## If you wish to use only some of the GPUs on your machine, you can do so like this:

mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])

with mirrored_strategy.scope():
        model = tf.keras.Model(inputs=inputs, outputs=x)
        model.compile(...)
        model.fit(...)