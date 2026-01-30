from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

python
import tensorflow as tf

inp = tf.keras.layers.Input((1,1))
mask = tf.keras.backend.not_equal(inp, 0)
gru = tf.keras.layers.GRU(1, name='my-output')(inp, mask=mask)

model = tf.keras.Model(inp, gru)

model.save('my-saved-model')
model2 = tf.keras.models.load_model('my-saved-model')