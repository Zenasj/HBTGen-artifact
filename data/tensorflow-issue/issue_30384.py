import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

inp = tf.keras.Input(batch_size=8, shape=(32, 32, 3))
tensor = tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3))(inp)
model = tf.keras.Model(inputs=inp, outputs=tensor)
def null_fn(y_true, y_pred):
    return tf.constant(0.)

model.compile('adam',loss=null_fn)
model.save('model.keras.tf', save_format='tf')
tf.keras.models.load_model('model.keras.tf', custom_objects={'null_fn': null_fn})