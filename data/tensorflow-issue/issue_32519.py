import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

ds = tf.data.TFRecordsDataset(...).shuffle(...)
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(...),
     ...
])

model.compile(...)
model.fit(...)