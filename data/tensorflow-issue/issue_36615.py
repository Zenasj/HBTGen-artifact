from tensorflow import keras

import tensorflow as tf

class Model(tf.keras.Model):

  def __init__(self):
    super(Model, self).__init__()
    self.table = tf.lookup.experimental.DenseHashTable(
        key_dtype=tf.int64,
        value_dtype=tf.int64,
        default_value=-1,
        empty_key=0,
        deleted_key=-1)

  @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.int64)])
  def call(self, input):
    return self.table.lookup(input)

m = Model()
tf.saved_model.save(m, '/tmp/test')

import tensorflow as tf

class Model(tf.keras.Model):

  def __init__(self):
    super(Model, self).__init__()
    self.table = tf.lookup.experimental.DenseHashTable(
        key_dtype=tf.int64,
        value_dtype=tf.int64,
        default_value=-1,
        empty_key=0,
        deleted_key=-1)

  @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.int64)])
  def call(self, input):
    return self.table.lookup(input)
  
  def insert(self, keys, values):
    return self.table.insert(keys, values)

m = Model()
m.insert([1,2,3,4], [4,3,2,1])
print(m([1,2,3,4]))
m.summary()
tf.saved_model.save(m, '/tmp/test')