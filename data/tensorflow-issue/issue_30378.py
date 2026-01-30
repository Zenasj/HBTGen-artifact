import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

python
inp = tf.keras.Input(batch_size=32, shape=(32, 32, 3))
tensor = tf.keras.layers.Conv2D(filters=16, kernel_size=3)(inp)
model = tf.keras.Model(inputs=inp, outputs=tensor)
model.add_loss(tf.keras.losses.mean_absolute_error(tensor, tensor + 1))
model.compile('adam')
tf.keras.experimental.export_saved_model(model, 'model.tf')

python
inp = tf.keras.Input(batch_size=32, shape=(32, 32, 3))
tensor = tf.keras.layers.Conv2D(filters=16, kernel_size=3)(inp)
model = tf.keras.Model(inputs=inp, outputs=tensor)
lbd = tf.keras.layers.Lambda(lambda i: tf.keras.losses.mean_absolute_error(i[0], i[1]))
model.add_loss(lbd([tensor, tensor + 1]))
model.compile('adam')
tf.keras.experimental.export_saved_model(model, 'model.tf')

python
inp = tf.keras.Input(batch_size=8, shape=(32, 32, 3))
tensor = tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3))(inp)
model = tf.keras.Model(inputs=inp, outputs=tensor)
lbd = tf.keras.layers.Lambda(lambda i: tf.keras.losses.mean_absolute_error(i[0], i[1]))
model.add_loss(lbd([tensor, tensor + 1]))
model.compile('adam')
model.save('model.keras.tf', save_format='tf')
tf.keras.models.load_model('model.keras.tf', custom_objects={'lambda': lbd})