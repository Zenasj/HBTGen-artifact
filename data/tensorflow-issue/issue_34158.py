import random
from tensorflow.keras import layers
from tensorflow.keras import optimizers

model.compile(optimizer=keras.optimizers.RMSprop(),
              loss='mean_squared_error',
              metrics=['mean_squared_error'])

model.compile(optimizer=keras.optimizers.RMSprop(),
              loss=keras.losses.MeanSquaredError(),
              metrics=[keras.metrics.MeanSquaredError()])

import tensorflow as tf
from tensorflow import keras
import numpy as np

# %% Make some sine wave data
num_time_series = 100
max_length = 200
num_points = 1
training_data = np.zeros((num_time_series, max_length, num_points))
np.random.seed(123)
tf.random.set_seed(123)
for i in range(num_time_series):
    # Make a sequence that doesn't fill the array so there's padding at the end
    length = np.random.randint(0.25 * max_length, .75 * max_length + 1)
    period = np.random.random() * 20 + 5
    shift = np.random.random()
    training_data[i, 0:length, 0] = np.sin(2 * np.pi / period * np.linspace(0, length - 1, length) + shift)


# %% Define the model
def make_model(use_mask):
    input_seq = keras.layers.Input(shape=(None, num_points))
    masked_input_seq = keras.layers.Masking(mask_value=0.0)(input_seq) if use_mask else input_seq
    gru = keras.layers.GRU(units=3, return_sequences=True)(masked_input_seq)
    output = keras.layers.Dense(units=1)(gru)
    model = keras.Model(input_seq, output)
    model.compile(optimizer=keras.optimizers.RMSprop(),
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])
    return model


# %% Train model with mask. Note that reported 'loss' and 'mean_squared_error' differ
make_model(True).fit(training_data[:, :-1, :],
                     training_data[:, 1:, :],
                     batch_size=10,
                     epochs=3,
                     verbose=1)


# %% Train model without mask. Note that reported 'loss' and 'mean_squared_error' match
make_model(False).fit(training_data[:, :-1, :],
                      training_data[:, 1:, :],
                      batch_size=10,
                      epochs=3,
                      verbose=1)

import tensorflow as tf
import numpy as np
y_true = np.array([[[0.], [1.], [1.]]])
x = np.array([[[1.], [0.], [1.]]])
model = tf.keras.Sequential([tf.keras.layers.Masking(mask_value=0., input_shape=(3, 1))])
model.compile(loss='mae', metrics=['mae'])
model.fit(x, y_true)  # Would intuitively expect mask to give loss of 0.5, but get 0.3333

import tensorflow as tf
import numpy as np
y_true = np.array([[0.], [1.]])
x = np.array([[1.], [0.]])
weights = np.array([1., 0.])
model = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=(1,))])
model.compile(loss='mae', metrics=['mae'])
model.fit(x, y_true, sample_weight=weights)  # Might expect to see 1.0, but get 0.5