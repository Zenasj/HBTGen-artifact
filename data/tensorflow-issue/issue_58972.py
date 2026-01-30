from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import numpy as np
from tensorflow.keras.layers import Dense, Dropout
import tensorflow as tf

input_length = 2
latent_dim = 512
output_length = 2

model = tf.keras.Sequential([
    Dense(latent_dim, activation='relu', input_shape=(input_length,)),
    Dropout(rate=0.5),
    Dense(units=latent_dim, activation='relu'),
    Dense(units=output_length),
  ])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.losses.MeanAbsoluteError()])

x_train = np.ones(shape=(150000, 2))
y_train = np.ones(shape=(150000, 2))

history = model.fit(x_train, y_train, epochs=2, batch_size=128)

model.save('saved_model/my_model')
pretrained = tf.keras.models.load_model('saved_model/my_model')