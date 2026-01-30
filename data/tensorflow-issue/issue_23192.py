import tensorflow as tf
from tensorflow.python.training.checkpointable.tracking import Checkpointable


class Model(Checkpointable):

  def __init__(self):
    self.variable = tf.get_variable("variable", [2, 2])
    self.dict = {
        "test": 1,
        "test2": 2,
    }
    self.dict_var = {
        "test": 1,
        "test2": tf.get_variable("dict_var", [2, 2])
    }
    self.dict_nested_var = {
        "test": 1,
        "test2": {
            "var": tf.get_variable("dict_nested_var", [2, 2])
        }
    }
    self.list_var = [tf.get_variable("list_var", [2, 2])]


print('SAVE')
with tf.Graph().as_default():

  s = tf.Session()

  m = Model()

  s.run(tf.global_variables_initializer())

  c = tf.train.Checkpoint(model=m)

  c.save('checkpoints/', session=s)

print('RESTORE')
with tf.Graph().as_default():

  s = tf.Session()

  m = Model()

  m.dict_new = {
      "test": 1,
      "test2": 2
  }

  c = tf.train.Checkpoint(model=m)

  status = c.restore('checkpoints/-1')

  status.assert_consumed().run_restore_ops(s)