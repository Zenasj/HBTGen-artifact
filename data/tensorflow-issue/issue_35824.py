from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
from tensorflow.keras.layers import Dense, Concatenate, RepeatVector, Activation, Dot, Bidirectional, Embedding, Input, SpatialDropout1D, LSTM, Dropout, Lambda, Conv1D, Attention, AdditiveAttention, GlobalAveragePooling1D, TimeDistributed, AveragePooling1D
from tensorflow.keras.models import Model
import numpy as np

def example_2():
    # Encode each timestep
    input_1 = Input(shape=(None,), dtype='int64', name="Input1")
    input_2 = Input(shape=(None,), dtype='int64', name="Input2")

    output = Concatenate([input_1, input_2])
    output = TimeDistributed(output)([input_1, input_2])

    model = Model([input_1, input_2], output)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model


input_1 = np.array([[1, 2, 3, 4, 5, 6, 7]])
input_2 = np.array([[1, 1, 1, 1, 1, 1, 1]])
y = np.array([1, 0, 1, 1, 0, 0, 1])

example_2().fit(input=[input_1, input_2], output=y)