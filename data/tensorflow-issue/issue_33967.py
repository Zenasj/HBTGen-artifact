import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import models

import csv
import os
from tensorflow import keras
import tensorflow as tf
import tensorflow_datasets as tfds
import time

datasets, info = tfds.load("imdb_reviews", as_supervised=True, with_info=True)

words = ['zero', 'one', 'two', 'three', 'four']
word_ids = list(range(5))
vocab_init = tf.lookup.KeyValueTensorInitializer(words, np.array(word_ids, dtype=np.int64))
table = tf.lookup.StaticVocabularyTable(vocab_init, 1)

def preprocess(X_batch, y_batch):
    X_batch = tf.strings.substr(X_batch, 0, 300)
    X_batch = tf.strings.regex_replace(X_batch, b"<br\\s*/?>", b" ")
    X_batch = tf.strings.regex_replace(X_batch, b"[^a-zA-Z']", b" ")
    X_batch = tf.strings.split(X_batch)
    return X_batch.to_tensor(default_value=b"<pad>"), y_batch

def encode_words(X_batch, y_batch):
    return table.lookup(X_batch), y_batch

train_set = datasets['train'].batch(32).map(preprocess)
train_set = train_set.map(encode_words).prefetch(1)

# save words as metadata to include in tensorboard callback
with open('metadata.tsv', 'w') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerows(zip(['id'], ['word']))
    writer.writerows(zip(word_ids, words))
    writer.writerows(zip([5], ['oov']))

root_logdir = os.path.join(os.curdir, 'my_logs')

def get_run_logdir():
    run_id = time.strftime('run_%Y_%m_%d-%H_%M_%S')
    return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir()

embed_size = 6
model = keras.models.Sequential([
    keras.layers.Embedding(6, embed_size, input_shape=[None], name='embedding'),
    keras.layers.GRU(6, return_sequences=True),
    keras.layers.GRU(6),
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# three methods tried:
# 1. shows word_ids only in tensorboard projector
#tensorboard_cb = keras.callbacks.TensorBoard(run_logdir, embeddings_freq=1)
# 2. str obj has not attribute keys
#tensorboard_cb = keras.callbacks.TensorBoard(run_logdir, embeddings_freq=1, embeddings_metadata='metadata.tsv')
# 3. unrecognized embedding layer name
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir, embeddings_freq=1, embeddings_metadata={'embedding': 'metadata.tsv'})

history = model.fit(train_set, epochs=1, callbacks=[tensorboard_cb])