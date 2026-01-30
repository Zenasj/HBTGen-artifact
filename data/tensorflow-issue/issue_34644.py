from tensorflow import keras
from tensorflow.keras import models

import tensorflow as tf
from tensorflow.keras import layers


def get_model(stateful=False, batch_size=None):
    n_channels = 1
    sample_size = 10
    n_units = 5

    inputs = layers.Input(batch_shape=(batch_size, sample_size, n_channels),
                          name='timeseries_inputs')
    y = layers.LSTM(units=n_units, input_shape=(sample_size, n_channels),
                    activation='tanh', recurrent_activation='sigmoid', return_sequences=False,
                    stateful=stateful)(inputs)

    y = layers.Dense(1)(y)

    model = tf.keras.models.Model(inputs=inputs, outputs=y)

    return model


stateless_model = get_model(stateful=False)

stateful_model = get_model(stateful=True, batch_size=1)

stateless_model.save('stateless.h5')
stateful_model.save('stateful.h5')

stateless_model.save('stateless.tf')
stateful_model.save('stateful.tf')