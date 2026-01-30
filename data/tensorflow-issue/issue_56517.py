import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import functools
import os

import numpy as np
import tensorflow as tf
from tensorflow.python.data.ops.options import AutoShardPolicy

NUM_TRAIN_SAMPLES = 1000
NUM_DEV_SAMPLES = 10

def get_dataset(split: str, batch_size: int, max_length: int = 64):
    # Number generator
    def generator():
        for i in range(num_samples):
            _dims = np.random.randint(low=1, high=max_length, size=1)
            x = np.zeros(_dims, dtype=np.int32) + np.random.randint(low=1, high=5000, size=1)
            y = np.zeros(_dims, dtype=np.int32) + np.random.randint(low=1, high=5000, size=1)
            yield {'sources': x, 'targets': y}

    assert split in ("train", "dev")
    is_training = split == "train"

    num_samples = NUM_TRAIN_SAMPLES if is_training else NUM_DEV_SAMPLES
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = AutoShardPolicy.DATA
    options.experimental_deterministic = False
    dataset = tf.data.Dataset.from_generator(
        lambda: generator(),
        output_signature={
            'sources': tf.TensorSpec(shape=(None,), dtype=tf.int32),
            'targets': tf.TensorSpec(shape=(None,), dtype=tf.int32)
        }
    ).with_options(options=options)

    dataset = dataset.padded_batch(batch_size=batch_size)

    def map_to_example(example):
        sources, targets = example['sources'], example['targets']
        return sources, targets

    dataset = dataset.map(map_to_example, num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        dataset = dataset.shuffle(buffer_size=256)

    return dataset.repeat().prefetch(tf.data.AUTOTUNE)


def main():
    max_length = 64

    train_batch_size = 64
    valid_batch_size = 16

    # noinspection PyArgumentEqualDefault
    strategy = tf.distribute.MirroredStrategy(devices=["GPU:0", "GPU:1"])
    with strategy.scope():
        optimizer = tf.keras.optimizers.Adam()
        # Model
        inputs = tf.keras.layers.Input(shape=(None,), dtype=tf.int32)
        x = inputs
        x = tf.keras.layers.Embedding(input_dim=5000,
                                      output_dim=64)(x)
        x = tf.keras.layers.Dense(5000)(x)
        model = tf.keras.Model(inputs=inputs, outputs=x)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer)
        model.summary()

        train_data = get_dataset(
            split="train",
            batch_size=train_batch_size,
            max_length=max_length
        )

        valid_data = get_dataset(
            split="dev",
            batch_size=valid_batch_size,
            max_length=max_length,
        )

        model.fit(
            train_data,
            epochs=200,
            steps_per_epoch=5000,
            validation_data=valid_data,
            validation_steps=3
        )


if __name__ == '__main__':
    main()