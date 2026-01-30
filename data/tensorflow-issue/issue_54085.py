import random
from tensorflow.keras import models

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

print(tf.config.list_physical_devices("GPU"))

if __name__ == "__main__":

    epoch = 10
    batch_size = 2000
    number_output = 3
    number_features = 5
    backward = 992
    number_node = 40

    def train_gen():
        for i in range(1000):
            yield np.random.random((batch_size, backward, number_features)),\
                  np.random.random((batch_size, number_output))

    train_dataset = tf.data.Dataset.from_generator(train_gen, output_types=(tf.float64, tf.float64))
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

    drop_ratio = 0.2
    model = Sequential([layers.Input(shape=(backward, number_features))])
    model.add(layers.LSTM(number_node, return_sequences=False))
    model.add(layers.Dropout(drop_ratio))
    model.add(layers.Dense(number_output, activation='sigmoid'))
    model.compile(optimizer='adam', loss='mse')
    model.summary()

    history = model.fit(train_dataset, epochs=epoch)