from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

from tensorflow.keras.layers import Input, Dense

i = Input(shape=(1, ), sparse=True)
o = Dense(1)(i)

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

text = ['sdsadfasd', 'wdfsdfg', 'sDFQERG', 'ASDFASDFG']
x = TfidfVectorizer(analyzer='char').fit_transform(text)
x.sort_indices()  # This is required or the keras model fails
y = pd.Series([1, 0, 1, 0])

inputs = tf.keras.layers.Input(shape=(x.shape[1],), name='dense_inputs')
outputs = tf.keras.layers.Dense(1)(inputs)
model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer="Adam", loss="mse")

# Fails
model.fit(x, y)
model.predict(x)

# Works
model.fit({'dense_inputs': x}, y.values)
model.predict(x)