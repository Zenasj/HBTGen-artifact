import tensorflow as tf

class DataDepInitInternal(DataDepInit):
  """
  initialized is internal variable as Bool
  w is internal variable as Tensor
  """
  def initialize(self, x):
    ctx = tf.distribute.get_replica_context()
    if ctx:
      n = ctx.num_replicas_in_sync * 1.0
      mean,*_ = ctx.all_reduce(tf.distribute.ReduceOp.SUM, [tf.reduce_mean(x, axis=[0, 1, 2], keepdims=True) / n])
    return mean

  def call(self, x, first=True):
    if self.initialized: # <- this condition and below assignment cause error.
      self.initialized.assign(True)
      self.w.assign(self.initialize(x))
    return x - self.w

class DataDepInitInternal(DataDepInit):
  """
  initialized is internal variable as Bool
  w is internal variable as Tensor
  """
  def initialize(self, x):
    ctx = tf.distribute.get_replica_context()
    if ctx:
      n = ctx.num_replicas_in_sync * 1.0
      mean,*_ = ctx.all_reduce(tf.distribute.ReduceOp.SUM, [tf.reduce_mean(x, axis=[0, 1, 2], keepdims=True) / n])
    return mean

  def call(self, x, first=True):
    if first:
      self.initialized.assign(True)
      self.w.assign(self.initialize(x))
    return x - self.w