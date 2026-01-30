from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

batch_size = 80
epochs = 171
num_classes = 10
import os
save_dir = 'model'
model_name = 'model.h5'
import keras as keras
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
img_rows, img_cols = x_train.shape[1], x_train.shape[2]

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

import keras as keras
model = keras.models.Sequential()
model.add(keras.layers.ThresholdedReLU(theta=0.3514439122821289))
model.add(keras.layers.LeakyReLU(alpha=0.4855740853866919))
model.add(keras.layers.AveragePooling2D(pool_size = (1, 2), padding='same'))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(num_classes, activation='softsign'))
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

model_path = os.path.join(save_dir, model_name)
model.save(model_path)