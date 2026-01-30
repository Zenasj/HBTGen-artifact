from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')])
model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(
    x_train,
    y_train,
    epochs=8,
    batch_size=128,
    validation_split=0.2)

test_loss, test_acc = model.evaluate(x_test, y_test)
print('model_test_acc:', test_acc)

model.save("my_saved_path")

saved_model = tf.keras.models.load_model("my_saved_path")

test_loss, test_acc = saved_model.evaluate(x_test, y_test)
print('saved_model_test_acc:', test_acc)