tf.config.experimental.list_physical_devices('GPU')

[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]

print(device_lib.list_local_devices())

tf.config.list_physical_devices('GPU')

[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]

tf.test.is_gpu_available()

True

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer

for device in tf.config.experimental.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(device, True)

max_features = 1000000
maxlen = 200
train_size=442598
updatedtrainsize = 5;

my_data = pd.read_csv('mydata.csv')
y = my_data["label"]
x = my_data["url"]
z = np.array(x)
w = np.array(y)
x_train = z[0:train_size]
x_val = z[train_size:]
y_train = w[0:train_size]
y_val = w[train_size:]

for x in range(len(y_train)): 
  if "good" in y_train[x]:
    y_train[x] = 0
  else:
    y_train[x] = 1

for x in range(len(y_val)): 
  if "good" in y_val[x]:
    y_val[x] = 0
  else:
    y_val[x] = 1

tokenizer = Tokenizer(filters='/-.+',
                      lower=True,
                      split=' ',
                      char_level=False,
                      oov_token='<OOV>')
tokenizer.fit_on_texts(x_train)
tokenizer.fit_on_texts(x_val)
word_index = tokenizer.word_index

x_train = tokenizer.texts_to_sequences(x_train)
x_val = tokenizer.texts_to_sequences(x_val)
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)
x_train = np.array(x_train).astype('float32')
x_val = np.array(x_val).astype('float32')
y_train = np.array(y_train).astype('float32')
y_val = np.array(y_val).astype('float32')

inputs = keras.Input(shape=(None,), dtype="int32")
x = layers.Embedding(max_features, 128)(inputs)
x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
x = layers.Bidirectional(layers.LSTM(64))(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)

model.compile("adam", "binary_crossentropy", metrics=["accuracy"])

# with tf.device("/GPU:0"):
model.fit(x_train, y_train, batch_size=32, epochs=2, validation_data=(x_val, y_val))