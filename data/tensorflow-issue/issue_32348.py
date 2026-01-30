import math
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import optimizers

from tensorflow import keras
from tensorflow.keras import layers

inputs = keras.Input(shape=(784,), name='digits')
x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
x = layers.Dense(64, activation='relu', name='dense_2')(x)
outputs = layers.Dense(10, activation='softmax', name='predictions')(x)
model = keras.Model(inputs=inputs, outputs=outputs, name='3_layer_mlp')

# Useless custom loss here
def custom_loss(y_true, y_pred):
    return keras.backend.mean(keras.backend.square(y_true - y_pred), axis=-1)

model.compile(loss=custom_loss, optimizer=keras.optimizers.RMSprop())

model.save("model/", save_format='tf')

# Here comes the bug:
new_model = keras.models.load_model('model/', custom_objects={'loss': custom_loss})

from tensorflow import keras
from tensorflow.keras import layers

inputs = keras.Input(shape=(784,), name='digits')
x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
x = layers.Dense(64, activation='relu', name='dense_2')(x)
outputs = layers.Dense(10, activation='softmax', name='predictions')(x)
model = keras.Model(inputs=inputs, outputs=outputs, name='3_layer_mlp')

# Useless custom loss here
def custom_loss(y_true, y_pred):
    return keras.backend.mean(keras.backend.square(y_true - y_pred), axis=-1)

model.save("model", save_format='tf')
model.compile(loss=custom_loss, optimizer=keras.optimizers.RMSprop())
# Here comes the bug (no bug)
new_model = keras.models.load_model('model', custom_objects={'loss': custom_loss})

class WeightedBinaryCrossentropy(tf.keras.losses.Loss):

    def __init__(self, pos_weight, name='WeightedBinaryCrossentropy'):
        super().__init__(name=name)
        self.pos_weight = pos_weight

    def call(self, y_true, y_pred):
        # For numerical stability clip predictions to log stable numbers
        y_pred = tf.keras.backend.clip(y_pred,
                                       tf.keras.backend.epsilon(),
                                       1 - tf.keras.backend.epsilon())
        # Compute weighted binary cross entropy
        wbce = y_true * -tf.math.log(y_pred) * self.pos_weight + (1 - y_true) * -tf.math.log(1 - y_pred)
        # Reduce by mean
        return tf.reduce_mean(wbce)