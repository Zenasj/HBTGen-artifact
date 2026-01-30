import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

class StringLayer(tf.keras.layers.Layer):
  def __init__(self):
    super(StringLayer, self).__init__()

  def call(self, inputs):
    return tf.strings.join([some_python_function(word) for word in tf.strings.split(tf.strings.as_string(inputs), sep=" " )], separator=" ")
    
  

#model = tf.keras.models.Sequential()
#model.add(tf.keras.Input(shape=(1,), dtype=tf.string))
#model.add(StringLayer())

with tf.device('GPU'):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.Input(shape=(1,), dtype=tf.string))
    model.add(StringLayer())

@tf.function
def mapp(string):
    return tf.strings.lower(string)

class StringLayer(tf.keras.layers.Layer):

    def __init__(self, seperator = " "):
        super(StringLayer, self).__init__()
        self.seperator = seperator

    def call(self, inputs):
        splits = tf.strings.split(inputs, sep=self.seperator)
        mapped = tf.map_fn(mapp,splits)
        joined = tf.strings.reduce_join(mapped, axis=2, separator=self.seperator)
        return joined