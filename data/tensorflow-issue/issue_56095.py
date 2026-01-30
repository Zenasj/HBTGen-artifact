import random
from tensorflow.keras import layers
from tensorflow.keras import models

import numpy as np
import os
import psutil
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, StringLookup, Dense
from tensorflow.keras import Model


vocabulary = list('ABCDEFGHIJKLMN')
seq_length = 800000
batch_size = 16


def build_model():
    # Build a simple model with only the StringLookup and activation
    input_layer = Input(shape=(seq_length,), dtype='string', name='input')

    output_layer = StringLookup(
        vocabulary=vocabulary,
        mask_token='',
        output_mode='multi_hot',
        name='string_lookup'
    )(input_layer)
    
    output_layer = Dense(1, activation='sigmoid')(output_layer)
    return Model(inputs=input_layer, outputs=output_layer)


def train_model(model):
    # Train the model with batch_size x seq_length batches
    x_arr = np.array([np.random.choice(vocabulary, size=(seq_length,)) for _ in range(batch_size)])
    y_arr = np.random.choice([0, 1], size=(batch_size,))

    model = build_model()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics='AUC')
    model.fit(x_arr, y_arr, batch_size=1, epochs=1)

    return model


def save_model(model):
    # Save the model and print the saved model file size & memory usage
    tf.keras.models.save_model(model, 'models/test/')
    print(os.path.getsize('models/test/saved_model.pb'), psutil.Process().memory_info().rss)

    
def load_model():
    return tf.keras.models.load_model('models/test/')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        save_model(train_model(build_model()))
    else:
        save_model(load_model())