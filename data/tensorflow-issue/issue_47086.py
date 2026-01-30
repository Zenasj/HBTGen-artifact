import tensorflow as tf

class Twice(tf.train.Checkpoint):
  def __init__(self):
    self._two = tf.Variable(2.0, name="two")

  @tf.function(input_signature=[tf.TensorSpec((None, None), tf.float32)])
  def __call__(self, x):
    return tf.multiply(x, self._two)

export_dir = "/tmp/twice"
tf.saved_model.save(Twice(), export_dir)

strategy = tf.distribute.MirroredStrategy()
with tf.Graph().as_default() as g:
  with strategy.scope():
    obj = tf.saved_model.load(export_dir)

  for v in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES):
    prefix = "NO INIT" if v.initializer is None else "ok init"
    print(prefix, v)

  init_op = tf.compat.v1.global_variables_initializer()