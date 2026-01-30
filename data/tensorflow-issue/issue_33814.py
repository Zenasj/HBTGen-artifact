from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import datetime
import os
import tensorflow as tf
import tensorflow_datasets as tfds


def preprocess_for_mnist(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


datasets, info = tfds.load(name='mnist',
                           with_info=True,
                           as_supervised=True,
                           shuffle_files=False)

strategy = tf.distribute.OneDeviceStrategy(device='/gpu:0')

train_dataset = datasets['train'].map(preprocess_for_mnist).batch(32)

log_dir = os.path.join(os.path.expanduser('~/tf_logs'),
                       datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32,
                               3,
                               activation='relu',
                               input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                       histogram_freq=0,
                                       profile_batch=2)
    ]

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])
    model.fit(train_dataset, epochs=2, steps_per_epoch=10, callbacks=callbacks)