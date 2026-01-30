from tensorflow.keras import models

converter = tf.lite.TFLiteConverter.from_saved_model('model_mnist.hd5')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

images = tf.cast(X_train, tf.float32)
mnist_ds = tf.data.Dataset.from_tensor_slices((images)).batch(1)
def representative_data_gen():
    for input_value in mnist_ds.take(100):
        yield[input_value]
converter.representative_dataset = representative_data_gen

tflite_quant_model = converter.convert()
with open('model_mnist_quant_uint8.tflite', 'wb') as f:
    f.write(tflite_quant_model)


interpreter = tf.lite.Interpreter(model_path='model_mnist_quant_uint8.tflite')
interpreter.allocate_tensors()

img = X_train[0] * 255
img = img.astype('uint8')
print(interpreter.get_input_details())
interpreter.set_tensor(interpreter.get_input_details()[0]['index'], np.expand_dims(img, axis=0))

# Put link here or attach to the issue.

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import numpy as np

from tensorflow.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

model = Sequential()
model.add(layers.Flatten(input_shape=X_train.shape[1:]))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()

model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train, epochs=2, batch_size=32, validation_split=0.2)

model.save('model_mnist')

converter = tf.lite.TFLiteConverter.from_saved_model('model_mnist')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

images = tf.cast(X_train, tf.float32)
mnist_ds = tf.data.Dataset.from_tensor_slices((images)).batch(1)
def representative_data_gen():
    for input_value in mnist_ds.take(100):
        yield[input_value]
converter.representative_dataset = representative_data_gen

tflite_quant_model = converter.convert()
with open('model_mnist_quant_uint8.tflite', 'wb') as f:
    f.write(tflite_quant_model)

interpreter = tf.lite.Interpreter(model_path='model_mnist_quant_uint8.tflite')
interpreter.allocate_tensors()

img = X_train[0] * 255
img = img.astype('uint8')
print(interpreter.get_input_details())
interpreter.set_tensor(interpreter.get_input_details()[0]['index'], np.expand_dims(img, axis=0))