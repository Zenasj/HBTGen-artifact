import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.config.set_visible_devices(gpus[:1], device_type='GPU')
log_dev_conf = tf.config.LogicalDeviceConfiguration(
    memory_limit=3*512
)
tf.config.set_logical_device_configuration(
    gpus[0],
    [log_dev_conf]) 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')

tf.keras.backend.clear_session()

Y_train = to_categorical(Y_train, num_classes=10, dtype=int)
Y_val = to_categorical(Y_val, num_classes=10, dtype=int)

batch_size = 2
data_generator = ImageDataGenerator(rescale=1.0/255.0)
data_generator = data_generator.flow(X_train, Y_train, batch_size=batch_size)

data_generator_val = ImageDataGenerator(rescale=1.0/255.0)
data_generator_val = data_generator_val.flow(X_val, Y_val, batch_size=batch_size)


model = Sequential()

model.add(Conv2D(128, (3, 3), input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(10, dtype='float32'))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer = 'Adam',
              metrics=['accuracy'])


# Train the model using the fit method
trained_model = model.fit(
    data_generator,
    validation_data=data_generator_val,
    steps_per_epoch=batch_size,
    epochs=5
)