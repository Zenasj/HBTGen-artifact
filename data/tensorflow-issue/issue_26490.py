from tensorflow import keras

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import datasets, layers, models

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

def customLossThatWorks():
    return tf.keras.losses.sparse_categorical_crossentropy

# def customLoss(y_true, y_pred):
#     return K.sparse_categorical_crossentropy(y_true, y_pred)

# def customLoss():
#     def loss(y_true,y_pred):
#         return K.sparse_categorical_crossentropy(y_true, y_pred)
#     return loss
    
# def customLoss():
#     def loss(y_true,y_pred):
#         return tf.keras.losses.sparse_categorical_crossentropy(y_true,y_pred)
#     return loss

# copied from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/losses.py#L867
def customLoss(y_true, y_pred, from_logits=False, axis=-1):
    return K.sparse_categorical_crossentropy(
      y_true, y_pred, from_logits=from_logits, axis=axis)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.compile(loss=customLoss, optimizer='adam', metrics=['accuracy', 'categorical_accuracy'])
model.fit(train_images, train_labels, epochs=1)

model.compile(loss=customLoss(), optimizer='adam', metrics=['accuracy', 'categorical_accuracy'])

model.compile(loss=customLoss, optimizer='adam', metrics=['accuracy', 'categorical_accuracy'])