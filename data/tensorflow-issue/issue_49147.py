from tensorflow import keras
from tensorflow.keras import layers

import sys
import time
import numpy as np
import tensorflow as tf

print(f'sys.version_info: {sys.version_info}')  # sys.version_info(major=3, minor=7, micro=7, releaselevel='final', serial=0)
print(f'tf.version.VERSION: {tf.version.VERSION}')  # 2.4.1
print(f'tf.version.GIT_VERSION: {tf.version.GIT_VERSION}')  # v2.4.0-49-g85c8b2a817f

steps = 50
batch_size = 16
input_w = 224
input_shape = (input_w, input_w, 3)

model_grouped = tf.keras.Sequential(layers=[
    tf.keras.layers.Input(shape=input_shape, batch_size=batch_size),
    tf.keras.layers.Conv2D(128, 1, padding='same'),
    tf.keras.layers.Conv2D(128, 3, padding='same', groups=8),  # 0
    tf.keras.layers.Conv2D(128, 3, padding='same', groups=8),  # 1
    tf.keras.layers.Conv2D(128, 3, padding='same', groups=8),  # 2
    tf.keras.layers.Conv2D(128, 3, padding='same', groups=8),  # 3
    tf.keras.layers.Conv2D(128, 3, padding='same', groups=8),  # 4
    tf.keras.layers.Conv2D(128, 3, padding='same', groups=8),  # 5
    tf.keras.layers.Conv2D(128, 3, padding='same', groups=8),  # 6
    tf.keras.layers.Conv2D(128, 3, padding='same', groups=8),  # 7
    tf.keras.layers.Conv2D(128, 3, padding='same', groups=8),  # 8
    tf.keras.layers.Conv2D(128, 3, padding='same', groups=8),  # 9
    tf.keras.layers.GlobalMaxPooling2D(),
    tf.keras.layers.Dense(2),
    tf.keras.layers.Activation('softmax')], name="Grouped")

model_regular = tf.keras.Sequential(layers=[
    tf.keras.layers.Input(shape=input_shape, batch_size=batch_size),
    tf.keras.layers.Conv2D(128, 1, padding='same'),
    tf.keras.layers.Conv2D(128, 3, padding='same'),  # 0
    tf.keras.layers.Conv2D(128, 3, padding='same'),  # 1
    tf.keras.layers.Conv2D(128, 3, padding='same'),  # 2
    tf.keras.layers.Conv2D(128, 3, padding='same'),  # 3
    tf.keras.layers.Conv2D(128, 3, padding='same'),  # 4
    tf.keras.layers.Conv2D(128, 3, padding='same'),  # 5
    tf.keras.layers.Conv2D(128, 3, padding='same'),  # 6
    tf.keras.layers.Conv2D(128, 3, padding='same'),  # 7
    tf.keras.layers.Conv2D(128, 3, padding='same'),  # 8
    tf.keras.layers.Conv2D(128, 3, padding='same'),  # 9
    tf.keras.layers.GlobalMaxPooling2D(),
    tf.keras.layers.Dense(2),
    tf.keras.layers.Activation('softmax')], name="Regular")

model_grouped.compile(optimizer=tf.optimizers.SGD(0.1, 0.8, True), loss=tf.keras.losses.BinaryCrossentropy())
model_regular.compile(optimizer=tf.optimizers.SGD(0.1, 0.8, True), loss=tf.keras.losses.BinaryCrossentropy())

images = np.full((2 * batch_size, input_w, input_w, 3), 0.5)
labels = np.full((2 * batch_size, 2), 1)

print("Warming up")
model_grouped.fit(images, labels, batch_size=batch_size)
model_regular.fit(images, labels, batch_size=batch_size)
print("Warmup completed")

images = np.full((steps * batch_size, input_w, input_w, 3), 0.5)
labels = np.full((steps * batch_size, 2), 1)

print("Training GROUPED model")
t0 = time.time()
model_grouped.fit(images, labels,
                  batch_size=batch_size)  # 50/50 [==============================] - 16s 317ms/step - loss: 0.0573
delta = time.time() - t0
print(f"Trained GROUPED in {delta: .3f} seconds")  # Trained GROUPED in  16.019 seconds

print("Training REGULAR model")
t0 = time.time()
model_regular.fit(images, labels,
                  batch_size=batch_size)  # 50/50 [==============================] - 19s 387ms/step - loss: 0.0000e+00
delta = time.time() - t0
print(f"Trained REGULAR in {delta: .3f} seconds")  # Trained REGULAR in  19.492 seconds

t0 = time.time()
model_grouped.predict(images, batch_size=batch_size, verbose=1)  # 50/50 [==============================] - 4s 78ms/step
print(f"Predicted GROUPED in {time.time() - t0: .3f} seconds")  # Predicted GROUPED in  4.148 seconds

t0 = time.time()
model_regular.predict(images, batch_size=batch_size,
                      verbose=1)  # 50/50 [==============================] - 7s 135ms/step
print(f"Predicted REGULAR in {time.time() - t0: .3f} seconds")  # Predicted REGULAR in  6.825 seconds

import sys
import time
import numpy as np
import tensorflow as tf

print(
    f'sys.version_info: {sys.version_info}')  # sys.version_info(major=3, minor=7, micro=7, releaselevel='final', serial=0)
print(f'tf.version.VERSION: {tf.version.VERSION}')  # 2.4.1
print(f'tf.version.GIT_VERSION: {tf.version.GIT_VERSION}')  # v2.4.0-49-g85c8b2a817f

steps = 50
batch_size = 16
input_w = 224
input_shape = (input_w, input_w, 3)

model_grouped = tf.keras.Sequential(layers=[
    tf.keras.layers.Input(shape=input_shape, batch_size=batch_size),
    tf.keras.layers.Conv2D(128, 1, padding='same'),
    tf.keras.layers.Conv2D(128, 3, padding='same', groups=8),  # 0
    tf.keras.layers.Conv2D(128, 3, padding='same', groups=8),  # 1
    tf.keras.layers.Conv2D(128, 3, padding='same', groups=8),  # 2
    tf.keras.layers.Conv2D(128, 3, padding='same', groups=8),  # 3
    tf.keras.layers.Conv2D(128, 3, padding='same', groups=8),  # 4
    tf.keras.layers.Conv2D(128, 3, padding='same', groups=8),  # 5
    tf.keras.layers.Conv2D(128, 3, padding='same', groups=8),  # 6
    tf.keras.layers.Conv2D(128, 3, padding='same', groups=8),  # 7
    tf.keras.layers.Conv2D(128, 3, padding='same', groups=8),  # 8
    tf.keras.layers.Conv2D(128, 3, padding='same', groups=8),  # 9
    tf.keras.layers.GlobalMaxPooling2D(),
    tf.keras.layers.Dense(2),
    tf.keras.layers.Activation('softmax')], name="Grouped")

model_regular = tf.keras.Sequential(layers=[
    tf.keras.layers.Input(shape=input_shape, batch_size=batch_size),
    tf.keras.layers.Conv2D(128, 1, padding='same'),
    tf.keras.layers.Conv2D(128, 3, padding='same'),  # 0
    tf.keras.layers.Conv2D(128, 3, padding='same'),  # 1
    tf.keras.layers.Conv2D(128, 3, padding='same'),  # 2
    tf.keras.layers.Conv2D(128, 3, padding='same'),  # 3
    tf.keras.layers.Conv2D(128, 3, padding='same'),  # 4
    tf.keras.layers.Conv2D(128, 3, padding='same'),  # 5
    tf.keras.layers.Conv2D(128, 3, padding='same'),  # 6
    tf.keras.layers.Conv2D(128, 3, padding='same'),  # 7
    tf.keras.layers.Conv2D(128, 3, padding='same'),  # 8
    tf.keras.layers.Conv2D(128, 3, padding='same'),  # 9
    tf.keras.layers.GlobalMaxPooling2D(),
    tf.keras.layers.Dense(2),
    tf.keras.layers.Activation('softmax')], name="Regular")

model_grouped.compile(optimizer=tf.optimizers.SGD(0.1, 0.8, True), loss=tf.keras.losses.BinaryCrossentropy())
model_regular.compile(optimizer=tf.optimizers.SGD(0.1, 0.8, True), loss=tf.keras.losses.BinaryCrossentropy())

images = np.full((2 * batch_size, input_w, input_w, 3), 0.5)
labels = np.full((2 * batch_size, 2), 1)

print("Warming up")
model_grouped.fit(images, labels, batch_size=batch_size)
model_regular.fit(images, labels, batch_size=batch_size)
print("Warmup completed")

images = np.full((steps * batch_size, input_w, input_w, 3), 0.5)
labels = np.full((steps * batch_size, 2), 1)

print("Training GROUPED model")
t0 = time.time()
model_grouped.fit(images, labels,
                  batch_size=batch_size)  # 50/50 [==============================] - 16s 317ms/step - loss: 0.0573
print(f"Trained GROUPED in {time.time() - t0: .3f} seconds")  # Trained GROUPED in  16.019 seconds

print("Training REGULAR model")
t0 = time.time()
model_regular.fit(images, labels,
                  batch_size=batch_size)  # 50/50 [==============================] - 19s 387ms/step - loss: 0.0000e+00
print(f"Trained REGULAR in {time.time() - t0: .3f} seconds")  # Trained REGULAR in  19.492 seconds

t0 = time.time()
model_grouped.predict(images, batch_size=batch_size, verbose=1)  # 50/50 [==============================] - 4s 78ms/step
print(f"Predicted GROUPED in {time.time() - t0: .3f} seconds")  # Predicted GROUPED in  4.148 seconds

t0 = time.time()
model_regular.predict(images, batch_size=batch_size,
                      verbose=1)  # 50/50 [==============================] - 7s 135ms/step
print(f"Predicted REGULAR in {time.time() - t0: .3f} seconds")  # Predicted REGULAR in  6.825 seconds

import sys
import time
import numpy as np
import tensorflow as tf

print(
    f'sys.version_info: {sys.version_info}')  # sys.version_info(major=3, minor=7, micro=7, releaselevel='final', serial=0)
print(f'tf.version.VERSION: {tf.version.VERSION}')  # 2.4.1
print(f'tf.version.GIT_VERSION: {tf.version.GIT_VERSION}')  # v2.4.0-49-g85c8b2a817f

steps = 50
batch_size = 16
input_w = 224
input_shape = (input_w, input_w, 3)

model_grouped = tf.keras.Sequential(layers=[
    tf.keras.layers.InputLayer(input_shape=input_shape, batch_size=batch_size),
    tf.keras.layers.Conv2D(128, 1, padding='same', use_bias=False),
    tf.keras.layers.Conv2D(128, 3, padding='same', use_bias=False, groups=8),  # 0
    tf.keras.layers.Conv2D(128, 3, padding='same', use_bias=False, groups=8),  # 1
    tf.keras.layers.Conv2D(128, 3, padding='same', use_bias=False, groups=8),  # 2
    tf.keras.layers.Conv2D(128, 3, padding='same', use_bias=False, groups=8),  # 3
    tf.keras.layers.Conv2D(128, 3, padding='same', use_bias=False, groups=8),  # 4
    tf.keras.layers.Conv2D(128, 3, padding='same', use_bias=False, groups=8),  # 5
    tf.keras.layers.Conv2D(128, 3, padding='same', use_bias=False, groups=8),  # 6
    tf.keras.layers.Conv2D(128, 3, padding='same', use_bias=False, groups=8),  # 7
    tf.keras.layers.Conv2D(128, 3, padding='same', use_bias=False, groups=8),  # 8
    tf.keras.layers.Conv2D(128, 3, padding='same', use_bias=False, groups=8),  # 9
    tf.keras.layers.GlobalMaxPooling2D(),
    tf.keras.layers.Dense(2),
    tf.keras.layers.Activation('softmax')], name="Grouped")

model_regular = tf.keras.Sequential(layers=[
    tf.keras.layers.InputLayer(input_shape=input_shape, batch_size=batch_size),
    tf.keras.layers.Conv2D(128, 1, padding='same', use_bias=False),
    tf.keras.layers.Conv2D(128, 3, padding='same', use_bias=False),  # 0
    tf.keras.layers.Conv2D(128, 3, padding='same', use_bias=False),  # 1
    tf.keras.layers.Conv2D(128, 3, padding='same', use_bias=False),  # 2
    tf.keras.layers.Conv2D(128, 3, padding='same', use_bias=False),  # 3
    tf.keras.layers.Conv2D(128, 3, padding='same', use_bias=False),  # 4
    tf.keras.layers.Conv2D(128, 3, padding='same', use_bias=False),  # 5
    tf.keras.layers.Conv2D(128, 3, padding='same', use_bias=False),  # 6
    tf.keras.layers.Conv2D(128, 3, padding='same', use_bias=False),  # 7
    tf.keras.layers.Conv2D(128, 3, padding='same', use_bias=False),  # 8
    tf.keras.layers.Conv2D(128, 3, padding='same', use_bias=False),  # 9
    tf.keras.layers.GlobalMaxPooling2D(),
    tf.keras.layers.Dense(2),
    tf.keras.layers.Activation('softmax')], name="Regular")

model_grouped.compile(optimizer=tf.optimizers.SGD(0.1, 0.8, True), loss=tf.keras.losses.BinaryCrossentropy())
model_regular.compile(optimizer=tf.optimizers.SGD(0.1, 0.8, True), loss=tf.keras.losses.BinaryCrossentropy())

images = np.full((2 * batch_size, input_w, input_w, 3), 0.5)
labels = np.full((2 * batch_size, 2), 1)

print("Warming up")
model_grouped.fit(images, labels, batch_size=batch_size)
model_regular.fit(images, labels, batch_size=batch_size)
print("Warmup completed")

images = np.full((steps * batch_size, input_w, input_w, 3), 0.5)
labels = np.full((steps * batch_size, 2), 1)

print("Training GROUPED model")
t0 = time.time()
model_grouped.fit(images, labels,
                  batch_size=batch_size)  # 50/50 [==============================] - 16s 317ms/step - loss: 0.0573
print(f"Trained GROUPED in {time.time() - t0: .3f} seconds")  # Trained GROUPED in  16.019 seconds

print("Training REGULAR model")
t0 = time.time()
model_regular.fit(images, labels,
                  batch_size=batch_size)  # 50/50 [==============================] - 19s 387ms/step - loss: 0.0000e+00
print(f"Trained REGULAR in {time.time() - t0: .3f} seconds")  # Trained REGULAR in  19.492 seconds

t0 = time.time()
model_grouped.predict(images, batch_size=batch_size, verbose=1)  # 50/50 [==============================] - 4s 78ms/step
print(f"Predicted GROUPED in {time.time() - t0: .3f} seconds")  # Predicted GROUPED in  4.148 seconds

t0 = time.time()
model_regular.predict(images, batch_size=batch_size,
                      verbose=1)  # 50/50 [==============================] - 7s 135ms/step
print(f"Predicted REGULAR in {time.time() - t0: .3f} seconds")  # Predicted REGULAR in  6.825 seconds