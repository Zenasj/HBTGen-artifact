import random
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import numpy as np
import tensorflow as tf
from tensorflow import keras

dataset_size = 3200
seq_len = 20
vocab_size = 10

X = np.random.randint(0, vocab_size, (dataset_size, seq_len))
Y = X

input_layer = keras.layers.Input(shape = (seq_len,), dtype = np.int32)
embedding_layer = keras.layers.Embedding(vocab_size, 5)(input_layer)
output_layer = keras.layers.Dense(vocab_size, activation = 'softmax')(embedding_layer)

model = keras.models.Model(inputs = input_layer, outputs = output_layer)
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = keras.optimizers.Adam(), sample_weight_mode = 'temporal')
model.fit(X, Y)