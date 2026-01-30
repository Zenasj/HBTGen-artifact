from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import tensorflow as tf
import contextlib
import sys

distributed=False
if len(sys.argv) > 1 and sys.argv[1] == "distributed":
    distributed=True
    print("asdf enabled distributed trainer")

class NoOpScope:
    def scope(self):
        return contextlib.suppress()

distribution_strategy = tf.distribute.MirroredStrategy() if distributed else NoOpScope()

with distribution_strategy.scope():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=[28, 28]))
    model.add(tf.keras.layers.Dense(300, activation="relu"))
    model.add(tf.keras.layers.Dense(100, activation="relu"))
    model.add(tf.keras.layers.Dense(10, activation="softmax"))
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

(x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_dev, x_train = x_train_full[:5000], x_train_full[5000:]
y_dev, y_train = y_train_full[:5000], y_train_full[5000:]
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(55000).repeat().batch(100)
dev_data = tf.data.Dataset.from_tensor_slices((x_dev, y_dev)).batch(100)

model.fit(train_data, 
          epochs=5,
          steps_per_epoch=55000/100,
          validation_data=dev_data)