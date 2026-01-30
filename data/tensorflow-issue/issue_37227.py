import random
from tensorflow import keras
from tensorflow.keras import optimizers

import numpy as np
import tensorflow as tf

from tensorflow.keras.losses import Loss

MINI_BATCH = 3
SPLITS = 2
BATCH = SPLITS * MINI_BATCH


class CustomLoss(Loss):
    def call(self, labels, outputs):

        @tf.function
        def fxn(inp):
            return tf.expand_dims(inp, 2)
        mapped = tf.map_fn(fxn, labels, dtype=tf.float32)
        return tf.reduce_sum(mapped)

def gen_batches():
    for _ in range(1000):
        outp = np.random.random((BATCH, 64))
        mask = np.random.random((SPLITS, MINI_BATCH, MINI_BATCH)) > .5
        mask = tf.cast(mask, tf.float32)
        yield outp, mask


if __name__ == "__main__":
    # Toggle to run as training (throws exception) vs merely iterating and calculating loss
    AS_MODEL = True

    dataset = tf.data.Dataset.from_generator(gen_batches, (tf.float32, tf.float32), output_shapes=([BATCH, 64], [SPLITS, MINI_BATCH, MINI_BATCH]))
    loss = CustomLoss()

    if AS_MODEL:
        inputs = tf.keras.Input(shape=(64,))
        out = inputs + 1
        model = tf.keras.Model(inputs=inputs, outputs=out)
        opt = tf.keras.optimizers.Nadam(.0005)
        model.compile(optimizer=opt, loss=loss)

        # This throws exception
        model.fit(dataset, epochs=1)

    else:
        # This runs fine
        for data in dataset.__iter__():
            tf.print(loss(data[1], data[0]))