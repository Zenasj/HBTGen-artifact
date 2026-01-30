from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import numpy as np
import tensorflow as tf
from tensorflow_model_optimization.python.core.quantization.keras import quantize
from tensorflow.python import keras
l = keras.layers

tf.config.run_functions_eagerly(True)

def layers_list():
  return [
      l.Conv2D(32, 5, padding='same', activation='relu',
               input_shape=image_input_shape(), activity_regularizer=tf.keras.regularizers.l2(l=0.0001), kernel_regularizer=tf.keras.regularizers.l2(l=0.0001)),
      l.MaxPooling2D((2, 2), (2, 2), padding='same'),
      # TODO(pulkitb): Add BatchNorm when transformations are ready.
      # l.BatchNormalization(),
      l.Conv2D(64, 5, padding='same', activation='relu', activity_regularizer=tf.keras.regularizers.l2(l=0.0001), kernel_regularizer=tf.keras.regularizers.l2(l=0.0001)),
      l.MaxPooling2D((2, 2), (2, 2), padding='same'),
      l.Flatten(),
      l.Dense(1024, activation='relu'),
      l.Dropout(0.4),
      l.Dense(10, activation='softmax')
  ]


def sequential_model():
  return keras.Sequential(layers_list())


def functional_model():
  """Builds an MNIST functional model."""
  inp = keras.Input(image_input_shape())
  x = l.Conv2D(32, 5, padding='same', activation='relu', activity_regularizer=tf.keras.regularizers.l2(l=0.0001), kernel_regularizer=tf.keras.regularizers.l2(l=0.0001))(inp)
  x = l.MaxPooling2D((2, 2), (2, 2), padding='same')(x)
  # TODO(pulkitb): Add BatchNorm when transformations are ready.
  # x = l.BatchNormalization()(x)
  x = l.Conv2D(64, 5, padding='same', activation='relu', activity_regularizer=tf.keras.regularizers.l2(l=0.0001), kernel_regularizer=tf.keras.regularizers.l2(l=0.0001))(x)
  x = l.MaxPooling2D((2, 2), (2, 2), padding='same')(x)
  x = l.Flatten()(x)
  x = l.Dense(1024, activation='relu')(x)
  x = l.Dropout(0.4)(x)
  out = l.Dense(10, activation='softmax')(x)

  return keras.models.Model([inp], [out])


def image_input_shape(img_rows=28, img_cols=28):
  if tf.keras.backend.image_data_format() == 'channels_first':
    return 1, img_rows, img_cols
  else:
    return img_rows, img_cols, 1

def preprocessed_data(img_rows=28,
                      img_cols=28,
                      num_classes=10):
  """Get data for mnist training and evaluation."""
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

  if tf.keras.backend.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
  else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')
  x_train /= 255
  x_test /= 255

  # convert class vectors to binary class matrices
  y_train = tf.keras.utils.to_categorical(y_train, num_classes)
  y_test = tf.keras.utils.to_categorical(y_test, num_classes)

  return x_train, y_train, x_test, y_test



model = functional_model() #sequential_model()
model.summary()
x_train, y_train, x_test, y_test = preprocessed_data()

model.compile(
    loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=500)
_, model_accuracy = model.evaluate(x_test, y_test, verbose=0)

print("Quantizing model")

quantized_model = quantize.quantize_model(model)
print(quantized_model.losses)
quantized_model.compile(
    loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
print(quantized_model.losses.numpy())

quantized_model.fit(x_train, y_train, batch_size=500)
_, quantized_model_accuracy = quantized_model.evaluate(
    x_test, y_test, verbose=0)