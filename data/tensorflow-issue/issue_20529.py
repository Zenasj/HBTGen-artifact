from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf

from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, UpSampling2D


class BatchCounter(tf.keras.layers.Layer):

        def __init__(self, name='batch_counter', **kwargs):
            super(BatchCounter, self).__init__(name=name, **kwargs)
            self.stateful = True
            self.batches = tf.keras.backend.variable(value=0, dtype='int32')

        def reset_states(self):
            tf.keras.backend.set_value(self.batches, 0)

        def __call__(self, y_true, y_pred):
            updates = [tf.keras.backend.update_add(self.batches, tf.keras.backend.variable(value=1, dtype='int32'))]
            self.add_update(updates)
            return self.batches


batch_size = 100
num_classes = 10
epochs = 1

# input image dimensions
img_rows, img_cols = 28, 28

# Data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1).astype('float32') / 255
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# Convolutional Encoder
input_img = Input(shape=(img_rows, img_cols, 1))
conv_1 = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
pool_1 = MaxPooling2D((2, 2), padding='same')(conv_1)
conv_2 = Conv2D(8, (3, 3), activation='relu', padding='same')(pool_1)
pool_2 = MaxPooling2D((2, 2), padding='same')(conv_2)
conv_3 = Conv2D(8, (3, 3), activation='relu', padding='same')(pool_2)
encoded= MaxPooling2D((2, 2), padding='same')(conv_3)

# Classification
flatten = Flatten()(encoded)
fc = Dense(128, activation='relu')(flatten)
softmax = Dense(num_classes, activation='softmax', name='classification')(fc)

# Decoder
conv_4 = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
up_1 = UpSampling2D((2, 2))(conv_4)
conv_5 = Conv2D(8, (3, 3), activation='relu', padding='same')(up_1)
up_2 = UpSampling2D((2, 2))(conv_5)
conv_6 = Conv2D(16, (3, 3), activation='relu')(up_2)
up_3 = UpSampling2D((2, 2))(conv_6)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='autoencoder')(up_3)

model = Model(inputs=input_img, outputs=[softmax, decoded])

model.compile(loss={'classification': 'categorical_crossentropy',
                    'autoencoder': 'binary_crossentropy'},
              loss_weights={'classification': 1.0,
                            'autoencoder': 0.5},
              optimizer='adam',
              metrics={'classification': 'accuracy', 'autoencoder': BatchCounter()})

history = model.fit(x_train,
          {'classification': y_train, 'autoencoder': x_train},
          batch_size=batch_size,
          epochs=epochs,
          validation_data= (x_test, {'classification': y_test, 'autoencoder': x_test}),
          verbose=1)

import tensorflow as tf
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Model

import numpy as np

class BatchCounter(tf.keras.layers.Layer):

        def __init__(self, name="batch_counter", **kwargs):
            super(BatchCounter, self).__init__(name=name, **kwargs)
            self.stateful = True
            self.batches = tf.keras.backend.variable(value=0, dtype="int32")

        def reset_states(self):
            tf.keras.backend.set_value(self.batches, 0)

        def __call__(self, y_true, y_pred):
            updates = [
                tf.keras.backend.update_add(
                    self.batches, 
                    tf.keras.backend.variable(value=1, dtype="int32"))]
            self.add_update(updates)
            return self.batches

class DummyGenerator(object):
    """ Dummy data generator. """

    def run(self):
        while True:
            yield np.ones((10, 1)), np.zeros((10, 1))

train_gen = DummyGenerator()
val_gen = DummyGenerator()

# Dummy model
inputs = Input(shape=(1,))
outputs = Dense(1)(inputs)
model = Model(inputs=inputs, outputs=outputs)
model.compile(loss="mse", optimizer="adam", metrics=[BatchCounter()])

model.fit_generator(
    train_gen.run(), 
    steps_per_epoch=5, 
    epochs=10, 
    validation_data=val_gen.run(), 
    validation_steps=5)

import tensorflow as tf
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Model

import numpy as np

class BatchCounter(tf.keras.layers.Layer):

        def __init__(self, name="batch_counter", **kwargs):
            super(BatchCounter, self).__init__(name=name, **kwargs)
            self.stateful = True
            self.batches = tf.keras.backend.variable(value=0, dtype="int32")

        def reset_states(self):
            tf.keras.backend.set_value(self.batches, 0)

        def __call__(self, y_true, y_pred):
            updates = [
                tf.keras.backend.update_add(
                    self.batches, 
                    tf.keras.backend.variable(value=1, dtype="int32"))]
            self.add_update(updates)
            return self.batches

# Dummy dataset
X = np.ones((50, 1))
y = np.zeros((50, 1))

# Dummy model
inputs = Input(shape=(1,))
outputs = Dense(1)(inputs)
model = Model(inputs=inputs, outputs=outputs)
model.compile(loss="mse", optimizer="adam", metrics=[BatchCounter()])

model.fit(X, y, batch_size=10, epochs=10, validation_data = (X, y))