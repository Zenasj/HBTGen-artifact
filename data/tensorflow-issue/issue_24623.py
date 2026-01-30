import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

3
model.load_weights("./test/trained.h5")

3
import numpy as np
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.layers import Dense

from files_functions import *

tf.enable_eager_execution()

def measure(y_true, y_pred):
    return tf.reduce_mean(y_pred - y_true)


class DemoNet(tf.keras.Model):
    def __init__(self):
        super(DemoNet, self).__init__()
        self.encoder_layer = Dense(128, input_shape=(128,), activation='tanh')

    def call(self, inputs, training=None, mask=None):
        """Run the model."""
        encoded = self.encoder_layer(inputs)

        return encoded


model = DemoNet()
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='mse',
              metrics=[measure])

training_input = np.ones((30000, 128)).astype(np.float32)
current_path = "./test/"
save_path = current_path
checkpoint = ModelCheckpoint(save_path + '/trained.h5', monitor='val_loss', verbose=0, save_best_only=True,
                             mode='min',
                             save_weights_only=True)
model.fit(training_input, training_input, epochs=1, batch_size=512, verbose=2, \
          callbacks=[checkpoint], validation_split=0.2)

del model

model = DemoNet()
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='mse',
              metrics=[measure])

testdata = np.ones((3000, 128))

model.load_weights("./test/trained.h5")
eval = model.evaluate(testdata, testdata, batch_size=512)

model.fit(training_input[:1], training_input[:1], epochs=1)

def DemoNet():
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Dense(128, input_shape=(128,), activation="tanh"))
    
    return model

def DemoNet():
    a = tf.keras.layers.Input(shape=(128,))
    b = tf.keras.layers.Dense(128)(a)
    
    model = tf.keras.models.Model(inputs=a, outputs=b)
    return model

class DemoModel(tf.keras.Model):
    def __init__(self):
        super(DemoModel, self).__init__()
        self.encoder_layer = Dense(128, input_shape=(128,), activation='tanh')

    def call(self, inputs):
        """Run the model."""
        encoded = self.encoder_layer(inputs)

        return encoded

def DemoNet():
    a = tf.keras.layers.Input(shape=(128,))
    b = DemoModel()(a)
    
    model = tf.keras.models.Model(inputs=a, outputs=b)
    return model