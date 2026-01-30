import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def preprocess(data):
  labels = list(dict(filter(lambda x: x[0] != 'vectorizer_input', data.items())).values())
  return data['vectorizer_input'], labels

tfdataset = tfdataset.map(preprocess)
vectorizer = tf.keras.layers.TextVectorization(max_tokens=4000, output_sequence_length=4)
vectorizer.adapt(tfdataset.map(lambda x, y: x))
model = tf.keras.Sequential(
    [
        vectorizer,
        tf.keras.layers.Embedding(4000, 64),
        tf.keras.layers.GlobalMaxPool1D(),
        tf.keras.layers.Dense(97, activation='sigmoid')
    ]
)