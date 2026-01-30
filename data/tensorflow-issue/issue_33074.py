import random
from tensorflow import keras

#!/usr/bin/env python3
  
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# UNCOMMENT THIS LINE AND YOU WILL GET AN ERROR
# tf.enable_eager_execution()

def tf_dataset():

    def data_gen():
        while True:
            yield np.random.rand(72, 96, 1), np.random.rand(18)

    types = (tf.float64, tf.float64)
    shapes = (tf.TensorShape([None, None, None]), tf.TensorShape([None]))
    dataset = tf.data.Dataset.from_generator(data_gen, types, output_shapes=shapes)
    dataset = dataset.batch(16)
    return dataset

dataset = tf_dataset()

model = tf.keras.Sequential()
# Notice how the input dimensions are mismatched
model.add(layers.InputLayer((96, 72, 1)))
model.add(layers.Conv2D(filters=32, kernel_size=4, strides=4, activation='relu'))
model.add(layers.Conv2D(filters=32, kernel_size=4, strides=4, activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(18, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy')
print(model.summary())
model.fit(dataset, steps_per_epoch=1, epochs=1)