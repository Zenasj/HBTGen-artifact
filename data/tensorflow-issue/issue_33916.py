import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers

## run.py ##
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.optimizers import Adam

#tf.keras.backend.set_image_data_format("channels_first")

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(filters=1, kernel_size=3, strides=2, input_shape=(224, 224, 3)))
model.add(tf.keras.layers.Conv2D(filters=1, kernel_size=3, strides=2))
model.add(tf.keras.layers.Conv2D(filters=1, kernel_size=3, strides=2))
model.add(tf.keras.layers.Conv2D(filters=1, kernel_size=3, strides=2))
model.add(tf.keras.layers.Conv2D(filters=1, kernel_size=3, strides=2))
model.add(tf.keras.layers.Conv2D(filters=1, kernel_size=3, strides=2))
model.add(tf.keras.layers.Conv2D(filters=1, kernel_size=2, strides=1, activation='sigmoid'))

model.compile( optimizer=Adam(), loss='binary_crossentropy')

model.summary()

gpu = tf.test.is_gpu_available()
print("GPU is available:", gpu)
assert(gpu)

batch_size = 10
x = np.random.random((batch_size, 224, 224, 3))
y = np.random.random((batch_size, 1, 1, 1))

print("x", x.shape)
print("y", y.shape)

y_pred = model.predict(x)
print("Predict successful: ", y_pred.shape)

print("Begin training with fit")
model.fit(x, y, epochs=10)
print("Fit successful")

print("Begin training with train_on_batch")
for i in range(10):
    model.train_on_batch(x, y)
print("On batch successful")

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
tf.config.experimental.set_memory_growth(physical_devices[0], True)