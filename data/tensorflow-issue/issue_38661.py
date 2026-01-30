import tensorflow as tf

class MyModule(tf.Module):
    def __init__(self, name):
        super(MyModule, self).__init__(name=name)

@tf.function
def graphcry():
    mod_inst1 = MyModule(name='inst1')
    mod_inst2 = MyModule(name='inst2')
    myscalar = tf.constant(83.2)
    with tf.name_scope('scalaragain'):
        tf.summary.scalar('scalaragain', data=myscalar)
    with tf.name_scope(mod_inst1.name + ' and ' + mod_inst2.name):
        tf.summary.scalar('myscalar', data=myscalar)

graphcry()

import tensorflow as tf

@tf.function
def graphcry():
    mod_inst1 = tf.Module(name='inst1')
    mod_inst2 = tf.Module(name='inst2')
    myscalar = tf.constant(83.2)  # just a random number
    with tf.name_scope('scalaragain'):
        tf.summary.scalar('scalaragain', data=myscalar)  # this works!
    with tf.name_scope(mod_inst1.name + ' and ' + mod_inst2.name):
        tf.summary.scalar('myscalar', data=myscalar)  # this returns the error above

graphcry()

import tensorflow as tf

@tf.function
def graphcry():
    inst1 = 'inst1'
    inst2 = 'inst2'
    myscalar = tf.constant(83.2)  # just a random number
    with tf.name_scope('scalaragain_scope'):
        tf.summary.scalar('scalaragain', data=myscalar)  # this works!
    with tf.name_scope(inst1 + ' and ' + inst2):
        tf.summary.scalar('myscalar', data=myscalar)  # this returns the error above

graphcry()

import tensorflow as tf


inst1 = 'inst1'
inst2 = 'inst2'

@tf.function
def graphcry():
    myscalar = tf.constant(83.2)  # just a random number
    with tf.name_scope('scalaragain_scope'):
        tf.summary.scalar('scalaragain', data=myscalar)  # this works!
    with tf.name_scope(inst1 + ' and ' + inst2):
        tf.summary.scalar('myscalar', data=myscalar)  # this returns the error above

graphcry()

import tensorflow as tf


def graphcry():
    inst1 = 'inst1'
    inst2 = 'inst2'
    myscalar = tf.constant(83.2)  # just a random number
    with tf.name_scope('scalaragain_scope'):
        tf.summary.scalar('scalaragain', data=myscalar)  # this works!
    with tf.name_scope(inst1 + ' and ' + inst2):
        tf.summary.scalar('myscalar', data=myscalar)  # this returns the error above

tf.function(graphcry)()