import random

import sys

import tensorflow as tf
import numpy as np

from hooks import InitHook

def input_fn():
  dataset = tf.data.Dataset.range(100)
  # Make sequence data
  dataset = dataset.map(lambda x: {'x': [x]})
  dataset = dataset.repeat(3)
  return dataset

def model_fn(features, labels, mode, params):
  seq = features['x']
  with tf.device('/gpu:0'):
    arr = np.random.rand(3000000, 400)
    var = tf.get_variable('big', arr.shape, trainable=False)
    emb = tf.nn.embedding_lookup(var, seq)
  logits = tf.layers.dense(emb, 1000)
  predictions = tf.greater(logits, 0.0)
  # Don't care about loss but have to provide something
  loss = tf.reduce_mean(logits)
  trainable_vars = tf.trainable_variables()
  global_step = tf.train.get_or_create_global_step()
  saveable_vars = trainable_vars + [global_step]
  def init_fn(scaffold, sess):
    sess.run(var.initializer, {var.initial_value: arr})
  saver = tf.train.Saver(var_list=saveable_vars)
  if mode == tf.estimator.ModeKeys.TRAIN:
    scaffold = tf.train.Scaffold(init_fn=init_fn, saver=saver)
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train_op = optimizer.minimize(loss, global_step=global_step)
    output_spec = tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss,
      scaffold=scaffold,
      train_op=train_op)
  elif mode == tf.estimator.ModeKeys.EVAL:
    hooks = [InitHook(var.initializer, {var.initial_value: arr})]
    ready_for_local_init_op = tf.constant([], dtype=tf.string)
    ready_op = tf.constant([], dtype=tf.string)
    scaffold = tf.train.Scaffold(init_fn=init_fn, saver=saver, ready_for_local_init_op=ready_for_local_init_op, ready_op=ready_op)
    output_spec = tf.estimator.EstimatorSpec(
      scaffold=scaffold,
      mode=mode,
      evaluation_hooks=hooks,
      loss=loss)
  elif mode == tf.estimator.ModeKeys.PREDICT:
    ready_for_local_init_op = tf.constant([], dtype=tf.string)
    ready_op = tf.constant([], dtype=tf.string)
    scaffold = tf.train.Scaffold(ready_for_local_init_op=ready_for_local_init_op, ready_op=ready_op, saver=saver)
    hooks = [InitHook(var.initializer, {var.initial_value: arr})]
    predictions = {
      'predictions': predictions
    }
    output_spec = tf.estimator.EstimatorSpec(
      mode=mode,
      predictions=predictions,
      scaffold=scaffold,
      prediction_hooks=hooks)
  return output_spec

tf.logging.set_verbosity(tf.logging.INFO)
estimator = tf.estimator.Estimator(model_fn=model_fn, config=tf.estimator.RunConfig())
eval_spec = tf.estimator.EvalSpec(input_fn=input_fn, start_delay_secs=0, throttle_secs=3)
if sys.argv[1] == 'bad':
  train_spec = tf.estimator.TrainSpec(input_fn=input_fn)
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
elif sys.argv[1] == 'good':
  train_spec = tf.estimator.TrainSpec(input_fn=input_fn, max_steps=100)
  estimator.evaluate(input_fn=input_fn)
results = estimator.predict(input_fn=input_fn)
for result in results:
  pass

class InitHook(training.SessionRunHook):
  def __init__(self, op, feed_dict):
    self.op = op
    self.feed_dict = feed_dict

  def after_create_session(self, session, coord):  # pylint: disable=unused-argument
    session.run(self.op, feed_dict=self.feed_dict)

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()