from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

from __future__ import print_function
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
import numpy as np


num_classes = 10
epochs = 1

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

def create_model():
  input_sequence = tf.keras.layers.Input(dtype='float32', shape=input_shape, name='input_sequence') 
  conv1 = Conv2D(32, kernel_size=(3, 3),
                   activation='relu')(input_sequence)
  conv2d = Conv2D(64, (3, 3), activation='relu')(conv1)
  conv2d = Dropout(0.25)(MaxPooling2D(pool_size=(2, 2))(conv2d))
  conv2d = Flatten()(conv2d)
  conv2d = Dropout(0.5)(Dense(128, activation='relu')(conv2d))
  output = Dense(num_classes, activation='softmax')(conv2d)
  model = tf.keras.models.Model(inputs=[input_sequence], outputs=output)
  model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
                optimizer=keras.optimizers.Adam(),
                metrics=['accuracy'])
  return model
  
model = create_model()
batch_size = 64
model.fit([np.array(x_train)], np.array(y_train),
  verbose=1,
  batch_size = batch_size,
  epochs=epochs,
  validation_data=([x_test], np.array(y_test)))
a = model.predict([x_test])
for x, y in zip(a[:20], y_test[:20]):
  print(x, np.argmax(x), y)

from __future__ import print_function
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
import numpy as np

num_classes = 10
epochs = 1

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

def create_model():
  input_sequence = tf.keras.layers.Input(dtype='float32', shape=input_shape, name='input_sequence') 
  conv1 = Conv2D(32, kernel_size=(3, 3),
                   activation='relu')(input_sequence)
  conv2d = Conv2D(64, (3, 3), activation='relu')(conv1)
  conv2d = Dropout(0.25)(MaxPooling2D(pool_size=(2, 2))(conv2d))
  conv2d = Flatten()(conv2d)
  conv2d = Dropout(0.5)(Dense(128, activation='relu')(conv2d))
  output = Dense(num_classes, activation='softmax')(conv2d)
  model = tf.keras.models.Model(inputs=[input_sequence], outputs=output)
  def custom_loss():
    def loss(y_true, y_predict):
      return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_predict)
    return loss
  model.compile(loss=custom_loss(),
                optimizer=keras.optimizers.Adam(),
                metrics=['accuracy'])
  return model
  
model = create_model()
batch_size = 64
model.fit([np.array(x_train)], np.array(y_train),
  verbose=1,
  batch_size = batch_size,
  epochs=epochs,
  validation_data=([x_test], np.array(y_test)))
a = model.predict([x_test])
for x, y in zip(a[:20], y_test[:20]):
  print(x, np.argmax(x), y)