from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import tensorflow as tf

cols = [
    tf.feature_column.indicator_column(
        tf.feature_column.sequence_categorical_column_with_vocabulary_list(
            "a", vocabulary_list=["one", "two"]
        )
    ),
    tf.feature_column.embedding_column(
        tf.feature_column.sequence_categorical_column_with_hash_bucket(
            "b", hash_bucket_size=10
        ),
        dimension=2,
    ),
]
input_layers = {
    "a": tf.keras.layers.Input(
        shape=(None, 1), sparse=True, name="a", dtype="string"
    ),
    "b": tf.keras.layers.Input(
        shape=(None, 1), sparse=True, name="b", dtype="string"
    ),
}

fc_layer, _ = tf.keras.experimental.SequenceFeatures(cols)(input_layers)
x = tf.keras.layers.GRU(32)(fc_layer)
output = tf.keras.layers.Dense(10)(x)

model = tf.keras.models.Model(input_layers, output)

model.compile(
    loss=tf.keras.losses.MSE,
    optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
    metrics=[tf.keras.metrics.categorical_accuracy],
)

tf.saved_model.save(model, "model")

import tensorflow as tf

cols = [
    tf.feature_column.sequence_numeric_column('a'),
]
input_layers = {
    'a':
        tf.keras.layers.Input(shape=(None, 1), sparse=True, name='a'),
}

fc_layer, _ = tf.keras.experimental.SequenceFeatures(cols)(input_layers)
x = tf.keras.layers.GRU(32)(fc_layer)
output = tf.keras.layers.Dense(10)(x)

model = tf.keras.models.Model(input_layers, output)

model.compile(
    loss=tf.keras.losses.MSE,
    optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
    metrics=[tf.keras.metrics.categorical_accuracy])

tf.saved_model.save(model, "model")