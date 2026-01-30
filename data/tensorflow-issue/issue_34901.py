from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

(x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz",
                                                      num_words=None,
                                                      skip_top=0,
                                                      maxlen=None,
                                                      seed=113,
                                                      start_char=1,
                                                      oov_char=2,
                                                      index_from=3)

x_train = pad_sequences(x_train, padding="post")
maxlen = x_train.shape[1]
vocab_size = x_train.max() + 1

model = tf.keras.Sequential([
  tf.keras.layers.Input(shape=(maxlen,), name="sequence"),
  tf.keras.layers.Embedding(vocab_size, 32, mask_zero=True, name="word_embedding"),
  tf.keras.layers.GlobalAveragePooling1D(name="doc_embedding"),
  tf.keras.layers.Dense(16, activation="relu", name="relu"),
  tf.keras.layers.Dense(1, activation="sigmoid", name="sigmoid")
], name="nn_classifier")
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
metrics = model.fit(x=x_train, y=y_train, batch_size=256, epochs=1)
model.save("model.h5")

tf.keras.models.load_model("model.h5")  # Failed.

model = tf.keras.Sequential([
  tf.keras.layers.Embedding(vocab_size, 32, input_length=maxlen, mask_zero=True, name="word_embedding"),
  tf.keras.layers.GlobalAveragePooling1D(name="doc_embedding"),
  tf.keras.layers.Dense(16, activation="relu", name="relu"),
  tf.keras.layers.Dense(1, activation="sigmoid", name="sigmoid")
], name="nn_classifier")
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
metrics = model.fit(x=x_train, y=y_train, batch_size=256, epochs=1)
model.save("model.h5")

model2 = tf.keras.models.load_model("model.h5")
model2.predict(x_train[:10]) # OK.