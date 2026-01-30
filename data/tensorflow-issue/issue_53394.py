import tensorflow as tf
from tensorflow import keras


x1 = keras.Input(shape=(2, 4, 4, 1))
x2 = keras.Input(shape=(2, 2, 4))
x = tf.einsum('bmhwf,bmoh->bmowf', x1, x2)
model = keras.Model(inputs=[x1, x2], outputs=x)
model = tf.keras.Model.from_config(model.get_config())