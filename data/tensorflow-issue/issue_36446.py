import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf
import numpy as np
base_model = tf.keras.applications.ResNet50V2(input_shape=(180, 320, 3), weights=None, include_top=False)
x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
output = tf.keras.layers.Dense(8)(x)
model = tf.keras.models.Model(inputs=base_model.input, outputs=output)
model.compile(optimizer='adam', loss='mse')
data = np.random.rand(2, 180, 320, 3)
labels = np.random.rand(2, 8)

model.fit(data, labels, validation_data=(data,labels), batch_size=2, epochs=30)