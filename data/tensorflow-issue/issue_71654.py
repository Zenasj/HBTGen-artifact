from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras import backend as K
import numpy as np
import os

# Clear session
tf.keras.backend.clear_session()
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Optionally, force use of CPU for debugging

# Define the LSTM model
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(20, 20), unroll=True, use_bias=False, recurrent_activation='sigmoid'),
    Dropout(0.2),
    LSTM(100, unroll=True, use_bias=False, recurrent_activation='sigmoid'),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train_seq,y_train_seq, epochs=100, batch_size=64, verbose=1)
# Make predictions
train_predict = model.predict(X_train_seq)
test_predict = model.predict(X_test_seq)
# Invert predictions to original scale