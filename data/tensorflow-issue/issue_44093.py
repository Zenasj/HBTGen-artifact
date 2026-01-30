from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow_datasets as tfds
import tensorflow as tf

# This is the only functional change from the example code.
tf.compat.v1.disable_eager_execution()

# Copied from the example code at:
# https://github.com/tensorflow/docs/blob/master/site/en/tutorials/distribute/keras.ipynb
datasets, info = tfds.load(name="mnist", with_info=True, as_supervised=True)
mnist_train, mnist_test = datasets["train"], datasets["test"]
strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync))

train_dataset = mnist_train.map(lambda im, l: ((tf.cast(im, tf.float32) / 255), l)).batch(64)

with strategy.scope():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(32, 3, activation="relu", input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(10),
        ]
    )

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=["accuracy"],
    )

model.fit(train_dataset, epochs=12)