from tensorflow.keras import layers
from tensorflow.keras import models

save.save_model(self, filepath, overwrite, include_optimizer, save_format,
                    signatures, options, save_traces)

import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense, Dropout

model = Sequential([
    LSTM(128,input_shape=(4,5)),
    Dropout(0,2),
    Dense(1)
])

model.compile(loss='mae', optimizer='adam')
model.summary()

#model.save('testmodel', save_traces=False)
model.save('testmodel', save_traces=True)