#-*- coding: utf-8 -*-
#File:


import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug

a = tf.placeholder(tf.float32, [10])
b = a + 1
c = b * 2

class Hook(tf.train.SessionRunHook):
    def before_run(self, _):
        return tf.train.SessionRunArgs(fetches=c)

class Hook2(tf.train.SessionRunHook):
    def before_run(self, _):
        return tf.train.SessionRunArgs(fetches=b)

sess = tf.Session()
sess = tf_debug.LocalCLIDebugWrapperSession(sess)

class SessionCreator():
    def create_session(self):
        return sess
final_sess = tf.train.MonitoredSession(session_creator=SessionCreator(), hooks=[Hook(), Hook2()])

final_sess.run(b, feed_dict={a:np.arange(10)})