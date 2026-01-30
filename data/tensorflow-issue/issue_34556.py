import random
from tensorflow import keras
from tensorflow.keras import optimizers

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class MyModel(tf.keras.Model):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.dense = layers.Dense(num_classes, activation='sigmoid')

    @tf.function
    def call(self, inputs):
        self.add_loss(tf.reduce_sum(inputs), inputs=True)
        return self.dense(inputs)

data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))
model = MyModel(num_classes=10)
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
              loss='categorical_crossentropy')
model.fit(data, labels, epochs=50)