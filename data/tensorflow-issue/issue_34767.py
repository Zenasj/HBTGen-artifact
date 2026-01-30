import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

import tensorflow as tf
import numpy as np

def Model_Functional_API():

    inputs = tf.keras.Input(shape=(3, 2))
    encoder = tf.keras.layers.LSTM(10,return_sequences=True)
    encoder_outputs = encoder(inputs)
    projection_layer = tf.keras.layers.Dense(2)
    preds = projection_layer(encoder_outputs)
    model = tf.keras.Model(inputs,preds)

    return model

def Model_Sequence():

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(10,return_sequences=True))
    model.add(tf.keras.layers.Dense(2))

    return model

# model = Model_Functional_API()
model = Model_Sequence()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mean_squared_error')

data_x = np.random.random([64,3,2])
data_y = np.random.random([64,3,2])

model.fit(data_x,data_y,batch_size=64,epochs=2)

model.save('saved_model', save_format='tf')
# model.save('saved_model.h5')

import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('saved_model')
# model = tf.keras.models.load_model('saved_model.h5')

data_x = np.random.random([64,3,2])
data_y = np.random.random([64,3,2])

model.fit(data_x,data_y,batch_size=64,epochs=2)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                  save_weights_only=False,
                                                  verbose=0,
                                                  save_freq=4542*10)

checkpoint_dir = os.path.dirname(checkpoint_path)
model = tf.keras.models.load_model(checkpoint_dir)