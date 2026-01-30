from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
import tensorflow_datasets as tfds
import os
from tensorflow.python.keras.callbacks import ModelCheckpoint

class UnFreezeWeight(tf.keras.callbacks.Callback):
    def __init__(self, freeze_before_epoch):
        super().__init__()
        self.freeze_before_epoch = freeze_before_epoch

    def on_epoch_begin(self, epoch, logs=None):
        if self.freeze_before_epoch != epoch:
            return

        # Unfreeze all weight.
        print('set trainable to True.')
        for layer in self.model.layers:
            layer.trainable = True


def _normalize_img(img, label):
    img = tf.cast(img, tf.float32) / 255.
    return img, label


def main():
    train_ds = tfds.load('mnist', split='train', as_supervised=True)
    train_ds = train_ds.batch(32)
    train_ds = train_ds.map(_normalize_img)

    valid_ds = tfds.load('mnist', split='test', as_supervised=True)
    valid_ds = valid_ds.batch(32)
    valid_ds = valid_ds.map(_normalize_img)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Freeze all weight.
    for layer in model.layers:
        layer.trainable = False
        
    # If I unfreeze at epoch index 0, the model will learning.
    # But if I un freeze after epoch index 1, the model won't training.
    freeze = UnFreezeWeight(1)

    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])

    model.fit(train_ds,
              validation_data=valid_ds,
              callbacks=[freeze],
              epochs=5)


if __name__ == '__main__':
    main()

freeze = UnFreezeWeight(10)

# Unfreeze all weight.
self.model.make_train_function(force=True)