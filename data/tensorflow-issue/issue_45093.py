import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions 

class Test(tf.Module):
  def __init__(self):
    self.log_likes_list = None
    self.i = tf.constant(0)

  @tf.function
  def __call__(self, samples):

    @tf.function
    def rnd():
        return tfd.Normal(0,1).sample()+ tfd.Normal(3,1).sample()
      
    if self.log_likes_list is None:
        self.log_likes_list = tf.TensorArray(tf.float32, size=samples) 

    def cond(x,i):
        return tf.less(i, samples) 

    def body(x,i):
        #x=x.write(i,tfm.reduce_sum(tfd.Normal(rnd(), 1).log_prob(0.4)))
        # AttributeError: 'TensorArray' object has no attribute 'mark_used'
        x.write(i,tfm.reduce_sum(tfd.Normal(rnd(), 1).log_prob(0.4))).mark_used()
        return x, i+1 

    self.log_likes_list, i = tf.while_loop(cond, body, [self.log_likes_list, self.i])

    self.log_likes = self.log_likes_list.stack()

    self.log_like = tfm.reduce_mean(self.log_likes)

    loss = self.log_like

    return loss


T= Test()
t= T(5)

T.log_likes, T.log_like, 
# the code in below is not running 
T.log_likes_list.stack()

import tensorflow_probability as tfp
tfd = tfp.distributions
tfm = tf.math

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfm = tf.math

class Test(tf.Module):
  def __init__(self, max_samples=100):
    # Variables to store things across tf.function boundaries (that are not returns/args)
    self.i = tf.Variable(0, dtype=tf.int64)
    self.max_samples = max_samples
    self.log_likes = tf.Variable(tf.zeros([max_samples]))
    self.log_like = tf.Variable(0.0)

  @tf.function(jit_compile=True) 
  def __call__(self, samples):

    def rnd():
        return tfd.Normal(0,1).sample()+ tfd.Normal(3,1).sample()

    # Read things from outside tf.function
    x = tf.TensorArray(tf.float32, size=tf.cast(samples, tf.int32))
    i = self.i.read_value()

    x = x.unstack(self.log_likes.read_value())
    while i < samples:
      # Always assign back to `x` after `x.write`
      x = x.write(tf.cast(i, tf.int32),
                  tfm.reduce_sum(tfd.Normal(rnd(), 1).log_prob(0.4)))
      i += 1

    log_likes = x.stack()
    log_like = tfm.reduce_mean(log_likes)
    loss = tfm.reduce_mean(log_like)

    # Put things back
    self.i.assign(i)
    padded = tf.pad(log_likes, [[0, self.max_samples - samples]])
    self.log_likes.assign(padded)
    self.log_like.assign(log_like)

    return loss


T= Test()
t = T(tf.constant(100, dtype=tf.int64))