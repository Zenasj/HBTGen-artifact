import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

model1 = tf.keras.Sequential([
    tf.keras.layers.Input(shape = (81,), dtype = 'uint8'), 
    tf.keras.layers.Lambda(tf.keras.backend.one_hot, arguments={'num_classes': 10}, output_shape=(81, 10)),
])

tf.keras.models.save_model(model1, './model')
model2 = tf.keras.models.load_model('./model', custom_objects={'one_hot' : tf.keras.backend.one_hot})
model2.summary()