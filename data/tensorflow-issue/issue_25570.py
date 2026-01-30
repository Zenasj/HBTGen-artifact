from tensorflow.keras import layers

import numpy as np
import tensorflow.keras as keras
model = keras.Sequential()
model.add(keras.layers.Conv2D(32, 8, 
        input_shape=(3, 128, 128), data_format="channels_first"))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(1))
model.compile(optimizer='adam', loss='mse')
x = np.ones((64, 3, 128, 128), dtype=np.float16)
y = np.ones((64, 1), dtype=np.float16)
model.fit(x, y)