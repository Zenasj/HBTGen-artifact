import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models

model_copy =  tf.keras.models.clone_model(model)
model_copy.build()
model_copy.set_weights(model.get_weights())

model = tf.keras.Sequential([
...
])

model_copy =  tf.keras.models.clone_model(model)
model_copy.build()
model_copy.set_weights(model.get_weights())