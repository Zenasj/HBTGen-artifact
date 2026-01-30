import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

i = tf.keras.layers.Input(shape=(10,))
x = tf.keras.layers.Dense(2)(i)
o = tf.keras.layers.Activation("softmax")(x)
m = tf.keras.Model(inputs=i, outputs=o)
m.save('test_model_tf', save_format="tf")
m2 = tf.keras.models.load_model("test_model_tf")
m2.summary()