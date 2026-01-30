from tensorflow import keras
from tensorflow.keras import models

import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

print(tf.__version__)

model = tf.keras.models.load_model("/home/amish/PycharmProjects/myproject/scripts/temp.h5")
model.summary()
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]
converter.experimental_new_converter = True  # Add this line

tflite_model = converter.convert()

go_backwards=True

model = tf.keras.Sequential()
model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))

#model.add(layers.LSTM(128, return_sequences=True)) #Works fine without Birdirectional
model.add(layers.Dense(20, activation='softmax'))

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

model.summary()

model.fit(x_train, y_train, batch_size=32, epochs=13, shuffle=True, validation_data=(x_test, y_test))

model = tf.keras.Sequential()
model.add(layers.Masking(input_shape=(50,36)))
model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
model.add(layers.Dense(20, activation='softmax'))

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

model.summary()

model.fit(x_train, y_train, batch_size=32, epochs=13, shuffle=True, validation_data=(x_test, y_test))