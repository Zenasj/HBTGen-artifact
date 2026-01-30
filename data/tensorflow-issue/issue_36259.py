from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class CustomLoss(tf.keras.losses.MeanSquaredError):
    pass


model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=(1,))])
model.compile(optimizer='sgd', loss=CustomLoss())
model.save('model')

# ValueError: Unknown loss function: CustomLoss
tf.keras.models.load_model('model', compile=True)