import tensorflow as tf

class Model:
  def __init__(self):
    self.v = None
    self.t = None

  @tf.function  # (1)
  def bar(self):
    print('tracing bar_g')
    return self.t  # (2)

  def foo(self):
    y = self.bar()
    self.v.assign_sub(1.0)
    
  @tf.function
  def loop(self):
    print('tracing loop')
    if self.t is None:
      self.t = tf.constant(1.0)
    if self.v is None:
      self.v = tf.Variable(1.0)
    self.foo()

m = Model()
m.loop()