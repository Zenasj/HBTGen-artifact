import math
import random
from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
import psutil
import os

if __name__ == '__main__':
    act_dim = 8

    model = tf.keras.Sequential([tf.keras.layers.Input(500, ),
                                 tf.keras.layers.Dense(500, activation="relu"),
                                 tf.keras.layers.Dense(act_dim, activation="softmax")])

    tf.random.set_seed(0)
    print("memory used:", psutil.Process(os.getpid()).memory_info().rss)

    for j in range(100):
        for i in range(1000):
            x = np.random.rand(500, 500)
            prob = model(x)

            dist = tfp.distributions.Categorical(probs=prob)
            a = dist.sample()
        print("memory used:", psutil.Process(os.getpid()).memory_info().rss)

import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
import numpy as np
import psutil
import os

if __name__ == '__main__':
    act_dim = 8

    model = tf.keras.Sequential([tf.keras.layers.Input(500, ),
                                 tf.keras.layers.Dense(500, activation="relu"),
                                 tf.keras.layers.Dense(act_dim, activation="softmax")])

    tf.random.set_seed(0)
    print("memory used:", psutil.Process(os.getpid()).memory_info().rss)

    for j in range(100):
        for i in range(1000):
            x = np.random.rand(500, 500).astype(np.float32)
            prob = model(x)

            a = tf.random.categorical(logits=tf.math.log(prob), num_samples=1)
        print("memory used:", psutil.Process(os.getpid()).memory_info().rss)

def categorical(logits, num_samples, dtype=None, seed=None):
  logits = tf.convert_to_tensor(logits, name="logits")
  seed1, seed2 = tf.compat.v1.get_seed(seed)

  dt = tf.float32
  batch = logits.shape.dims[0].value
  num_classes = logits.shape.dims[1].value
  if num_classes == 0:
    # Delegates to native op to raise the proper error.
    return tf.random.categorical(
        logits, num_samples, seed=seed1, seed2=seed2, output_dtype=dtype)
  # u ~ Uniform[0.0, 1.0)
  u = tf.random.uniform(
      shape=[batch, num_samples],
      seed=seed1, seed2=seed2, dtype=dt)
  # for numerical stability
  max_logit = tf.math.reduce_max(logits, axis=1, keepdims=True)
  logits = logits - max_logit
  pdf = tf.cast(tf.math.exp(logits), dtype=dt)  # not normalized
  cdf = tf.math.cumsum(pdf, axis=1)
  cdf_last= cdf[:, -1:]
  u = u * cdf_last
  if num_samples == 0 or batch == 0:
    # A tf.searchsorted bug workaround
    return tf.zeros([batch, num_samples])
  else:
    return tf.searchsorted(cdf, u, side="right")  # upper_bound