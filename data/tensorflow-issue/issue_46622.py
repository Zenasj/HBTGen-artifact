from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

batch_size = 112
epochs = 119
num_classes = 10
import os
save_dir = 'model'
model_name = 'test971_trained_model.h5'
import tensorflow.keras as keras
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
img_rows, img_cols = x_train.shape[1], x_train.shape[2]
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
model = keras.models.Sequential()
model.add(keras.layers.BatchNormalization(momentum = 0.4639004933194679,epsilon=0.6515653837017596))
model.add(keras.layers.PReLU(alpha_initializer='Zeros'))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(num_classes))
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
model_path = os.path.join(save_dir, model_name)
model.save(model_path)