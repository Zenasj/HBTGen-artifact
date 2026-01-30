from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

3
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, MaxPool2D
model = Sequential([
    Conv2D(16,(3,3),padding='same', input_shape=(1,28,28),data_format='channels_first'),
    MaxPooling2D((3,3), data_format='channels_first')
])
opt = tf.keras.optimizers.Adam(learning_rate=0.005)
model.compile(optimizer=opt,
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy(),
              tf.keras.metrics.MeanAbsoluteError()]
              )
print(model.metrics)