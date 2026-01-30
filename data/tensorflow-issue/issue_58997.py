from tensorflow.keras import layers
from tensorflow.keras import models

# Simple model that illustrate the problem:

# First- create the model and save it. ( Relayed on: https://iq.opengenus.org/conv2d-in-tf/)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D, Flatten, Dense

fashion = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255


model = Sequential()
model.add(Conv2D(32, (3, 3),input_shape=(28, 28, 1)))
# model.add(MaxPooling2D((2, 2)))
# model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)


model.save('model.h5')


# -----------------------------------------------------------------------------
# Then: 
# load the model and run it few times:
model= tf.keras.models.load_model('model.h5')
print(model.predict(test_images))

# Gives different predictions


# You might need to remove cache files from one run to another to notice the differences: 
# sudo rm -rf ~/.nv/

print(model.predict(test_images))
import subprocess
subprocess.run('sudo rm -rf ~/.nv/',shell=True)
print(model.predict(test_images))