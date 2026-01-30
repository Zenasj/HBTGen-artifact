import random

# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend, layers, models

x = layers.Input(shape=(4,))
y = layers.Dense(1, activation='softmax')(x)
model = models.Model(inputs=x, outputs=y)
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

x_data = np.random.random((10, 4))
y_data = np.random.random((10,))
model.fit(x_data, y_data, epochs=5, batch_size=10)
model.save('./model', save_format='tf')  # Loading this will raise an error
model.save('./model.h5', save_format='h5')  # Loading this will work

# This block will load the SavedModel without compiling and will 
# perform the inference raising an error
backend.clear_session()
model2 = models.load_model('./model', compile=False)
model2.predict(x_data, batch_size=10)

# This block will load the Keras saved model without compiling and will 
# perform the inference working as expected
backend.clear_session()
model3 = models.load_model('./model.h5', compile=False)
model3.predict(x_data, batch_size=10)