import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, GlobalMaxPooling1D, Conv1D, BatchNormalization, Activation, concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import Constant
import numpy as np

vocab_size = 200_000
vocab_dim = 100
length1 = 50
length2 = 100
length = 100
num_samples = 100_000
conv_dim = 128
window_sizes = [2, 3, 4, 5]
num_classes = 2000

input_features = [Input(shape=(length1,), dtype="int32"), Input(shape=(length2,), dtype="int32")]
hidden_tensors = []
    
for feature in input_features:
    embeddings = Embedding(vocab_size, vocab_dim)(feature)

    for window_size in window_sizes:
        convoluted = Conv1D(conv_dim, window_size)(embeddings)
        normed = BatchNormalization()(convoluted)
        activated = Activation("relu")(normed)
        pooled = GlobalMaxPooling1D()(activated)
        hidden_tensors.append(pooled)
    
hidden = Dropout(0.5)(concatenate(hidden_tensors))
outputs = Dense(num_classes, activation='softmax')(hidden)

model = Model(inputs=input_features, outputs=outputs)
model.compile(loss="sparse_categorical_crossentropy", optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

features1 = np.random.randint(0, vocab_size, size=(num_samples, length1), dtype=np.int32)
features2 = np.random.randint(0, vocab_size, size=(num_samples, length2), dtype=np.int32)
labels = np.random.randint(0, num_classes, size=num_samples)

model.fit([features1, features2], labels, epochs=1, batch_size=256)

tf.keras.utils.set_random_seed(1234)
tf.config.experimental.enable_op_determinism()