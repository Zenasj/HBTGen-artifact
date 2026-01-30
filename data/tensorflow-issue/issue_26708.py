from tensorflow.keras import layers

run_model = tf.function(lambda x: model(x))
concrete_func = run_model.get_concrete_function(
      tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
converter = tf.lite.TFLiteConverter.from_concrete_function(concrete_func)
tflite_model = converter.convert()
open("/tmp/mnist_fashion/converted_model.tflite", "wb").write(tflite_model)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import keras
import numpy as np


def get_data_set(num_images):
  """Returns the training and testing data base."""
  fashion_mnist = keras.datasets.fashion_mnist
  (train_images, train_labels), (test_images, test_labels) = (
      fashion_mnist.load_data())
  train_images = train_images / 255.0
  test_images = test_images / 255.0

  num_images = min(num_images, len(train_images))
  train_images = train_images[:num_images]
  train_labels = train_labels[:num_images]
  return train_images, train_labels


def build_model(train_images, train_labels, epochs=5):
  """Build the model."""
  model = keras.Sequential([
      keras.layers.Flatten(input_shape=(28, 28)),
      keras.layers.Dense(128, activation='relu'),
      keras.layers.Dense(10, activation='softmax')
  ])
  model.compile(
      optimizer='adam',
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy'])
  model.fit(train_images, train_labels, epochs)
  return model


def main():
  export_dir = '/tmp/mnist_fashion'

  # Build the model.
  train_images, train_labels = get_data_set(num_images=100)
  model = build_model(train_images, train_labels, epochs=1)
  tf.saved_model.save(model, export_dir) # not needed for this code

  # Convert the model directly.
  run_model = tf.function(lambda x: model(x))
  concrete_func = run_model.get_concrete_function(
      tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
  converter = tf.lite.TFLiteConverter.from_concrete_function(concrete_func)
  tflite_model = converter.convert()
  open("/tmp/mnist_fashion/converted_model.tflite", "wb").write(tflite_model)


if __name__ == '__main__':
  main()

model = tf.saved_model.load(export_dir)
concrete_func = model.signatures[
    tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
concrete_func.inputs[0].set_shape([1, 28, 28])
converter = tf.lite.TFLiteConverter.from_concrete_function(concrete_func)
tflite_model = converter.convert()
open("/tmp/mnist_fashion/converted_model.tflite", "wb").write(tflite_model)