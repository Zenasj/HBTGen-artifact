import numpy as np
import math
import random
from tensorflow.keras import layers
from tensorflow.keras import optimizers

from __future__ import absolute_import, division, print_function, unicode_literals
import os, time, sys, numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (Conv2D, Conv3D, Dense)


@tf.function
def loss_fn(y_pred, y_true):
    return tf.reduce_mean(tf.math.square(y_pred - y_true))

@tf.function
def train_step(dist_inputs):
    def step_fn(inputs):
        inputs, labels = inputs

        # tf.print("in", tf.shape(inputs), "out", tf.shape(labels), output_stream=sys.stdout)
        with tf.GradientTape() as tape:
            out = model(inputs)
            loss_value = loss_fn(out, labels)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return loss_value

    per_example_losses = strategy.experimental_run_v2(step_fn, args=(dist_inputs,))
    mean_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_example_losses, axis=None)
    return mean_loss


if __name__ == "__main__":

    BATCH_SIZE_PER_SYNC = 4
    logdir = os.path.join('logs/test')
    strategy = tf.distribute.MirroredStrategy()
    num_gpus = strategy.num_replicas_in_sync
    global_batch_size = BATCH_SIZE_PER_SYNC * num_gpus
    print('num GPUs: {}, global batch size: {}'.format(num_gpus, global_batch_size))

    # fake data ------------------------------------------------------
    fakea = np.random.rand(global_batch_size, 10, 200, 200, 128).astype(np.float32)
    targets = np.random.rand(global_batch_size, 200, 200, 14)

    # tf.Dataset ------------------------------------------------------
    def gen():
        while True:
            yield (fakea, targets)

    dataset = tf.data.Dataset.from_generator(gen,
        (tf.float32, tf.float32),
        (tf.TensorShape(fakea.shape), tf.TensorShape(targets.shape)))

    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dist_dataset = strategy.experimental_distribute_dataset(dataset)

    # Model ------------------------------------------------------
    training = True
    with strategy.scope():
        # Model
        va = keras.Input(shape=(10, 200, 200, 128), dtype=tf.float32, name='va')
        x = Conv3D(64, kernel_size=3, strides=1, padding='same')(va)
        x = Conv3D(64, kernel_size=3, strides=1, padding='same')(x)
        x = Conv3D(64, kernel_size=3, strides=1, padding='same')(x)
        x = tf.reduce_max(x, axis=1, name='maxpool')  # [ΣK, 128]
        b = Conv2D(14, kernel_size=3, padding='same')(x)
        model = keras.Model(inputs=va, outputs=b, name='net')
        optimizer = keras.optimizers.RMSprop()
    model.summary()

    # TRAIN ---------------------------------------------------------
    writer = tf.summary.create_file_writer(logdir)

    num_steps = 100
    num_epoches = 100
    global_step = 0

    with strategy.scope():
        iterator = iter(dist_dataset)
        with writer.as_default():
            for epoch in range(num_epoches):
                for step in range(num_steps):

                    if global_step == 0 or 5 < global_step < 8:
                        tf.summary.trace_on(graph=True, profiler=True)

                    start = time.time()
                    loss_value = train_step(next(iterator))
                    duration = time.time() - start

                    prefix = 'Ep {:02d}/{:02d} | step {:02d} '.format(epoch + 1, num_epoches, step)
                    suffix = '| {:.3f} sec/step | loss: {:.3f} '.format(duration, float(loss_value))
                    print(prefix + suffix)

                    tf.summary.scalar("loss", loss_value, step=global_step)

                    if global_step == 0 or 5 < global_step < 8:
                        tf.summary.trace_export(name="model_trace", step=global_step, profiler_outdir=logdir)
                    writer.flush()
                    global_step += 1