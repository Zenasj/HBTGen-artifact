from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

data = {"numbers": [111, 222, 333, 444], "strings": ["a", "b", "c", "d"]}
dataset = tf.data.Dataset.from_tensor_slices(data).batch(2)

number_encoding = tf.keras.layers.IntegerLookup()
number_encoding.adapt(dataset.map(lambda x: x["numbers"]))

string_encoding = tf.keras.layers.StringLookup()
string_encoding.adapt(dataset.map(lambda x: x["strings"]))

string_model = tf.keras.Sequential([
  string_encoding,
  tf.keras.layers.Embedding(string_encoding.vocabulary_size(), 4)
])

number_model = tf.keras.Sequential([
  number_encoding,
  tf.keras.layers.Embedding(number_encoding.vocabulary_size(), 4)
])

print(number_model(next(iter(dataset.map(lambda x: x["numbers"]))))) # works

print(string_model(next(iter(dataset.map(lambda x: x["strings"]))))) #fails

tmp = string_encoding(next(iter(dataset.map(lambda x: x["strings"])))) # same as above, but works
print(tf.keras.layers.Embedding(string_encoding.vocabulary_size(), 4)(tmp)) # works