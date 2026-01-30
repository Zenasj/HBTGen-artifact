import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

with tf.device("/device:CPU:0"):
  ...

from tensorflow import keras
(X_train, y_train), (X_test, y_test) = keras.datasets.imdb.load_data(num_words=10000)
(X_valid, X_test) = X_test[:12500], X_test[12500:]
(y_valid, y_test) = y_test[:12500], y_test[12500:]
word_index = keras.datasets.imdb.get_word_index()
X_train_trim = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=500)
X_test_trim = keras.preprocessing.sequence.pad_sequences(X_test, maxlen=500)
X_valid_trim = keras.preprocessing.sequence.pad_sequences(X_valid, maxlen=500)
model = keras.models.Sequential()
model.add(keras.layers.Embedding(input_dim=10000, output_dim=10))
model.add(keras.layers.SimpleRNN(32))
model.add(keras.layers.Dense(1, "sigmoid"))

model.compile(loss='binary_crossentropy', optimizer="adam", metrics=["accuracy"])

history = model.fit(X_train_trim, y_train,epochs=10, batch_size=128, validation_data=(X_valid_trim, y_valid))

with tf.device("/device:CPU:0"):
    history = model.fit(X_train_trim, y_train,epochs=10, batch_size=128, validation_data=(X_valid_trim, y_valid))