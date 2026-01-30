from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

path = './checkpoint/model'

model = tf.keras.Sequential([
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(100)
])

model(tf.zeros([1, 100]))

checkpoint = tf.train.Checkpoint(model=model)
manager = tf.train.CheckpointManager(checkpoint, directory=path, max_to_keep=5)

manager.save()
print('saved1')
manager.save()  # here an error occurs
print('saved2')