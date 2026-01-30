import random

py
import os

import numpy as np
import tensorflow as tf


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


print(tf.__version__)


class Config:
    def __init__(self):
        self.units = 10
        self.n_classes = 2
        self.drop_rate = 0.5


class Model:
    def __init__(self, cfg):
        self.cfg = cfg
        self.mlp = tf.layers.Dense(cfg.units)
        self.resize = tf.layers.Dense(cfg.n_classes)

    def predict(self, x):
        z = self.mlp(x)
        z = tf.layers.dropout(z, rate=self.cfg.drop_rate,
                              training=self.cfg.training)
        z = self.resize(z)
        return z


cfg = Config()
# training = tf.placeholder_with_default(False, (), 'mode')
training = False
cfg.training = training

model = Model(cfg)


def _cond(x, i):
    return tf.less(i, 20)


def _body(x, i):
    y = model.predict(x)
    dy_dx = tf.gradients(y, x)[0]
    x = dy_dx
    return x, i+1


x = tf.placeholder(tf.float32, (None, 3))
y = model.predict(x)
xx, ind = tf.while_loop(_cond, _body, [x, 0])

sess = tf.Session()
sess.run(tf.global_variables_initializer())

val = sess.run(xx, feed_dict={x: np.random.random((1, 3))})
print(val)

sess.close()