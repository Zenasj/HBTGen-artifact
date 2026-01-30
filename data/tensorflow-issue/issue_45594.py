import random

import tensorflow as tf  # v2.4.0
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers


def sample_generator(nb_samples):

    for i in range(nb_samples):
        l = np.random.randint(6, 20)
        yield np.random.rand(l, 8), np.random.rand(1, 1)

    # One example for bucket (1, 5)
    yield np.random.rand(3, 8), np.random.rand(1, 1)


def sample_len(sample, *_):
    return tf.shape(sample)[0]


nb_replica = max(1, len(tf.config.experimental.list_physical_devices('GPU')))
assert nb_replica > 1, f'Number of GPUs must be >1 got {nb_replica}'

dataset = tf.data.Dataset.from_generator(
    lambda: sample_generator(500),
    output_types=(tf.float32, tf.float32),
    output_shapes=((None, 8), (None, 1))
)
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
dataset = dataset.with_options(options)

boundaries = [5, 10]
batch_sizes = [i * nb_replica for i in range(len(boundaries) + 1)]

bucketing = tf.data.experimental.bucket_by_sequence_length(
    sample_len,
    bucket_boundaries=boundaries,
    bucket_batch_sizes=batch_sizes,
    drop_remainder=True
)

dataset = dataset.apply(bucketing).repeat()

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    inputs = layers.Input(shape=(None, 8))
    x = inputs
    x = layers.LSTM(16)(x)
    x = layers.Dense(1)(x)
    model = keras.Model(inputs=inputs, outputs=x)
    model.compile(loss='mse')

model.fit(
    dataset,
    epochs=2,
    steps_per_epoch=100,
)

layers.Embedding(
    input_dim=SIZE_VOCAB,
    output_dim=EMBED_DIM,
    mask_zero=True,
    input_length=MAX_SEQ_LEN,
),