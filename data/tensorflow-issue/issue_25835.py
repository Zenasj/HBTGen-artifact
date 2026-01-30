from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

#!/usr/bin/env python
import tensorflow as tf
from tensorflow import keras

print(tf.__version__)

(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0


def create_model():
  model = tf.keras.models.Sequential([
    keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation=tf.nn.softmax)
  ])

  model.compile(optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.sparse_categorical_crossentropy,
                metrics=['accuracy'])

  return model


model = create_model()

checkpoint_path = "log_file/cp.bin"


model.fit(train_images, train_labels,  epochs=10, verbose=0,
          validation_data=(test_images, test_labels))  # pass callback to training
# model.save_weights(checkpoint_path, save_format='h5')
model.save_weights(checkpoint_path, save_format='tf')

model = create_model()
loss, acc = model.evaluate(test_images, test_labels)
print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

model.load_weights(checkpoint_path)
loss, acc = model.evaluate(test_images, test_labels)
print("trained model, accuracy: {:5.2f}%".format(100 * acc))