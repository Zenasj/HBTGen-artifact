from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_datasets as tfds
# tfds.disable_progress_bar()
(train_data, test_data), info = tfds.load(
    'imdb_reviews/subwords8k', 
    split = (tfds.Split.TRAIN, tfds.Split.TEST), 
    with_info=True, as_supervised=True)
encoder = info.features['text'].encoder
padded_shapes = ([None],())
train_batches = train_data.shuffle(1000).padded_batch(10, padded_shapes = padded_shapes, drop_remainder=True)
test_batches = test_data.shuffle(1000).padded_batch(10, padded_shapes = padded_shapes, drop_remainder=True)

embedding_dim=16

model = keras.Sequential([
    layers.Embedding(encoder.vocab_size, embedding_dim,mask_zero=True),
    layers.Bidirectional(tf.keras.layers.LSTM(32)),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_batches,
    epochs=10,
    validation_data=test_batches, validation_steps=20,verbose=2)

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_datasets as tfds
# tfds.disable_progress_bar()
(train_data, test_data), info = tfds.load(
    'imdb_reviews/subwords8k',
    split = (tfds.Split.TRAIN, tfds.Split.TEST),
    with_info=True, as_supervised=True)
encoder = info.features['text'].encoder
padded_shapes = ([None],())
train_batches = train_data.shuffle(1000).padded_batch(10, padded_shapes = padded_shapes, drop_remainder=True)
test_batches = test_data.shuffle(1000).padded_batch(10, padded_shapes = padded_shapes, drop_remainder=True)

embedding_dim=16

model = keras.Sequential([
    layers.Embedding(encoder.vocab_size, embedding_dim,mask_zero=True),
    layers.Bidirectional(tf.keras.layers.LSTM(32)),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_batches,
    epochs=10,
    validation_data=test_batches, validation_steps=20)

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_datasets as tfds
# tfds.disable_progress_bar()
(train_data, test_data), info = tfds.load(
    'imdb_reviews/subwords8k',
    split = (tfds.Split.TRAIN, tfds.Split.TEST),
    with_info=True, as_supervised=True)
encoder = info.features['text'].encoder
padded_shapes = ([None],())
train_batches = train_data.shuffle(1000).padded_batch(10, padded_shapes = padded_shapes, drop_remainder=True)
test_batches = test_data.shuffle(1000).padded_batch(10, padded_shapes = padded_shapes, drop_remainder=True)

embedding_dim=16

model = keras.Sequential([
    layers.Embedding(encoder.vocab_size, embedding_dim,mask_zero=False),
    # layers.Masking(mask_value=0),
    layers.Bidirectional(tf.keras.layers.LSTM(32)),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_batches,
    epochs=10,
    validation_data=test_batches, validation_steps=20)

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_datasets as tfds
# tfds.disable_progress_bar()
(train_data, test_data), info = tfds.load(
    'imdb_reviews/subwords8k',
    split = (tfds.Split.TRAIN, tfds.Split.TEST),
    with_info=True, as_supervised=True)
encoder = info.features['text'].encoder
padded_shapes = ([None],())
# train_batches = train_data.shuffle(1000).padded_batch(10, padded_shapes = padded_shapes, drop_remainder=True)
# # test_batches = test_data.shuffle(1000).padded_batch(10, padded_shapes = padded_shapes, drop_remainder=True)
train_batches = train_data.shuffle(1000).padded_batch(10, padded_shapes = padded_shapes)
test_batches = test_data.shuffle(1000).padded_batch(10, padded_shapes = padded_shapes)
embedding_dim=16

model = keras.Sequential([
    layers.Embedding(encoder.vocab_size, embedding_dim,mask_zero=False),
    # layers.Masking(mask_value=0),
    layers.Bidirectional(tf.keras.layers.LSTM(32)),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_batches,
    epochs=10,
    validation_data=test_batches, validation_steps=20)