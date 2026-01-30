import random
from tensorflow import keras
from tensorflow.keras import optimizers

#!/usr/bin/env python3
import sys
import tensorflow as tf
# Otherwise nothing works, and it really sucks, but is declared in the docs
multi_worker_strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

def main():
    batch_size = 12
    features_shape = 372, 558, 3
    labels = 10
    sample = tf.random.uniform(features_shape)

    def with_shape(t, shape):
        t = tf.squeeze(t)
        t.set_shape(shape)
        return t

    ds_train = tf.data.Dataset.from_tensors([sample]).map(lambda s: (s, tf.ones((labels,)))) \
        .repeat().batch(batch_size).map(lambda s, l: (with_shape(s, (batch_size,) + features_shape),
                                                      with_shape(l, (batch_size, labels))))
    ds_val = tf.data.Dataset.from_tensors([sample]).map(lambda s: (s, tf.ones((labels,)))) \
        .repeat().batch(batch_size).take(10).map(
        lambda s, l: (with_shape(s, (batch_size,) + features_shape), with_shape(l, (batch_size, labels))))
    with multi_worker_strategy.scope():
        model = tf.keras.applications.DenseNet121(
            weights=None, input_shape=features_shape, classes=labels)
        model.build((batch_size,) + features_shape)
        model.summary()
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
        cross_entropy = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
        model.compile(optimizer=optimizer, loss=cross_entropy, metrics=["accuracy"])
    model.fit(ds_train, validation_data=ds_val, epochs=1, steps_per_epoch=100)


if __name__ == "__main__":
    sys.exit(main())