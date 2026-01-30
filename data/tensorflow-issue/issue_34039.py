import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

#!/usr/bin/env python3
import sys
import tensorflow as tf

def main():
    strategy = tf.distribute.MirroredStrategy()
    batch_size = 12
    features_shape = 372, 558, 3
    labels = 10
    sample = tf.random.uniform(features_shape)

    def batch_print(b, l):
        tf.print("shape", b.shape, tf.shape(b))
        tf.print(b[10])  # <<< crash here
        return b, l

    ds_train = tf.data.Dataset.from_tensors([sample]).map(lambda s: (tf.squeeze(s), tf.ones((labels,)))) \
        .repeat().batch(batch_size, drop_remainder=True).map(batch_print)
    ds_val = tf.data.Dataset.from_tensors([sample]).map(lambda s: (tf.squeeze(s), tf.ones((labels,)))) \
        .repeat().batch(batch_size, drop_remainder=True).take(10)

    import tensorflow_core.python.keras.backend
    original_input = tensorflow_core.python.keras.layers.Input

    def create_input(*args, **kwargs):
        return original_input(*args, batch_size=batch_size, **kwargs)

    # monkey-patch the input layer to ensure the fixed tensor shape
    tensorflow_core.python.keras.layers.Input = create_input

    with strategy.scope():
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