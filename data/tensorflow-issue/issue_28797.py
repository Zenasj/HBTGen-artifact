from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

import tensorflow as tf
import tensorflow_datasets as tfds

mnist_train, mnist_test = tfds.load(name='mnist', split=[tfds.Split.TRAIN, tfds.Split.TEST], as_supervised=True)

strategy = tf.distribute.MirroredStrategy()

def scale(image, label):
  image = tf.cast(image, tf.float32)
  image /= 255

  return image, label

train_dataset = mnist_train.map(scale).shuffle(1000).batch(256)
test_dataset = mnist_test.map(scale).batch(256)

with strategy.scope():
  model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(10, activation='softmax')
  ])

  model.compile(loss='sparse_categorical_crossentropy',
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['accuracy'])

model.fit(train_dataset, validation_data=test_dataset, epochs=10)