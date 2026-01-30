import numpy as np
import math
import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

from tensorflow.compat.v1.keras.layers import Dense, Dropout, concatenate
from tensorflow.keras.layers import LSTM, Input, concatenate
from tensorflow.keras.initializers import glorot_uniform, Constant
from tensorflow.keras.models import Model
from tensorflow.keras import activations
import tensorflow as tf

DS_SIZE=4
SEQ_LENGTH=5
MAX_SEQ_LENGTH=5


def prepare_data(subjects, seq_lengths, max_seq_length):
    # Define the sub sequence lenghts
    seq_lengths = np.random.randint(max_seq_length, size=(subjects))
    
    # Get the values
    seq_values = np.random.randint(100, size=(sum(seq_lengths)), dtype=np.int32)

    # Create a nested ragged tensor, of shape [subjects, (seq_length), (0...max_seq_length)]
    X = tf.expand_dims(tf.RaggedTensor.from_row_lengths(values=seq_values, row_lengths=seq_lengths), 2)
    
    Y = tf.math.reduce_sum(X, axis=(2, 1))

    return (X, Y)
    
X, Y = prepare_data(DS_SIZE, SEQ_LENGTH, MAX_SEQ_LENGTH)

print(f'Tensorflow version: {(tf.version.GIT_VERSION, tf.version.VERSION)}')
print(f'Input data:\nX.shape = {X.shape}\nX={X}')

from tensorflow.keras.layers import Lambda

inputs = Input(shape=(None,1), name="Input", ragged=True)
lstm = LSTM(5, return_sequences=False, name='LSTM')(inputs)
predictor = Dense(1, activation=activations.linear, name='Predictor')(lstm)

model = Model(inputs=inputs, outputs=predictor)
model.summary()

model.compile(
    loss='mse',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.2),
    metrics=[tf.keras.metrics.MeanSquaredError()],
    run_eagerly=True  # Enable eager mode!
)

history = model.fit(
    X,
    Y,
    validation_split = 0.1,
    epochs = 1,
    batch_size = 2,
)