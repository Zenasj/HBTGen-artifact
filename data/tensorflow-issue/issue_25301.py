from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

trained_model.save_weights(output_weights)

tf.keras.backend.clear_session()
tf.keras.backend.set_learning_phase(0)    # This is the important part
eval_model = model_build_function()
eval_model.load_weights(output_weights, by_name=True)

import tensorflow as tf
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.losses as losses
from tensorflow.keras.datasets import mnist

(train_x, train_y), _ = mnist.load_data()

from tensorflow.keras.layers import *
from tensorflow.keras.models import *
regularizers = tf.keras.regularizers

reg_weight = 0.00001

model = Sequential()
model.add(Reshape((28, 28, 1), input_shape=(28, 28)))
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28,28,1),
                 kernel_regularizer=regularizers.l1(reg_weight)))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l1(reg_weight)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l1(reg_weight)))
model.add(Dense(10, activation='softmax', kernel_regularizer=regularizers.l1(reg_weight)))

model.compile(optimizer=optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

train_x = train_x[:100, :, :]
train_y = train_y[:100]
model.fit(train_x, tf.keras.utils.to_categorical(train_y), batch_size=64, epochs=2, verbose=1)
model.save("/tmp/25301.h5")
converter = tf.lite.TFLiteConverter.from_keras_model_file("/tmp/25301.h5")
tfl = converter.convert()
print(len(tfl))