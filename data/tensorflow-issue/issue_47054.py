from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf   # issue was reproduced with tf.__version__ == 2.4.1

model = tf.keras.Sequential([
    tf.keras.layers.Reshape((-1, 784)),
    tf.keras.layers.Lambda(lambda x: tf.divide(tf.cast(x, tf.float32), 255.)),
    tf.keras.layers.Dense(256, activation='relu', bias_initializer=tf.initializers.zeros),
    tf.keras.layers.Dense(10, activation='softmax',)
])

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
model.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, validation_data=(x_test, y_test))
model.save('test')

import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Reshape((-1, 784)),
    tf.keras.layers.Lambda(lambda x: tf.divide(tf.cast(x, tf.float32), 255.)),
    tf.keras.layers.Dense(256, activation='relu', bias_initializer=tf.initializers.Zeros()),
    tf.keras.layers.Dense(10, activation='softmax',)
])

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
model.compile('adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, validation_data=(x_test, y_test))
model.save('test')

@keras_export('keras.initializers.get')
def get(identifier):
  if identifier is None:
    return None
  if isinstance(identifier, dict):
    return deserialize(identifier)
  elif isinstance(identifier, six.string_types):
    identifier = str(identifier)
    return deserialize(identifier)
  elif callable(identifier):
    return identifier
  else:
    raise ValueError('Could not interpret initializer identifier: ' +
                     str(identifier))

@keras_export('keras.initializers.get')
def get(identifier):
  if identifier is None:
    return None
  if isinstance(identifier, dict):
    return deserialize(identifier)
  elif isinstance(identifier, six.string_types):
    identifier = str(identifier)
    return deserialize(identifier)
  elif callable(identifier):
    if inspect.isclass(identifier):  # Additional check copied from the snippet above
      identifier = identifier()
    return identifier
  else:
    raise ValueError('Could not interpret initializer identifier: ' +
                     str(identifier))