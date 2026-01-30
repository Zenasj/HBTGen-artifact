from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence

from tensorflow.keras.layers import (
    RNN,
    Input,
    Dense
)

from tensorflow.contrib.rnn import PhasedLSTMCell

K.clear_session()

with tf.variable_scope('test', reuse=tf.AUTO_REUSE):
    t_series_dim = 6
    output_dim = 1

    # Inputs here
    timestamps = Input(shape=(None, t_series_dim,), name='t_input')
    feature_signals = Input(shape=(None, t_series_dim,), name='ts_input')

    # Layer configuration
    ## Time-series input
    time_input = (timestamps, feature_signals)
    
    cell = PhasedLSTMCell(256)
    ts_x = RNN(cell, return_sequences=False)(time_input)
    
    # Output layer
    logits = Dense(output_dim, name='logits', activation='sigmoid')(ts_x)

    # Stick it all together inside a model...
    model = Model(
        inputs=[timestamps, feature_signals],
        outputs=[logits]
    )

    # Define an optimiser and metrics, and compile the model
    model.compile(
        optimizer='adam',
        loss=['binary_crossentropy'],
        metrics=['binary_accuracy']
    )

timestamps = Input(shape=(None, 1), name='t_input')