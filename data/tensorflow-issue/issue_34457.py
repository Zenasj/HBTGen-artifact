import random
from tensorflow import keras
from tensorflow.keras import optimizers

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

# Model from tf Keras overview
model = tf.keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(32,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# Data generator
def get_dataset(batch_size, num_batches):
    """ Get random tf.data dataset with given batch size and number of batches. """

    def _generator():
        for i in range(num_batches):
            data = np.random.random((batch_size, 32))
            labels = np.random.random((batch_size, 10))

            yield data, labels
            
    dataset = tf.data.Dataset.from_generator(
        _generator,
        (tf.float32, tf.float32),
        ((batch_size, 32), (batch_size, 10))
    )
    
    return dataset

# Create datasets
dataset = get_dataset(batch_size=32, num_batches=1024)
val_dataset = get_dataset(batch_size=32, num_batches=128)

# Train
model.fit(dataset, epochs=10,
          validation_data=val_dataset)