import tensorflow as tf
import numpy as np
import sys

class RunHook(tf.train.SessionRunHook):
  def before_run(self, run_context):
    return tf.train.SessionRunArgs(fetches=['var:0'])

  def after_run(self, run_context, run_values):
    print(run_values.results)
    sys.exit(0)

def model_fn(features, labels, mode, params):
  var = tf.get_variable(
    initializer=tf.constant([1.0], dtype=tf.float32),
    name="var",
    dtype=tf.float32,
    trainable=True,
  )
  loss = tf.identity(var)
  opt = tf.train.AdamOptimizer(0.001)
  global_step = tf.train.get_or_create_global_step()
  train_op = opt.minimize(loss, global_step=global_step)
  return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

strategy = tf.distribute.MirroredStrategy()
session_config = tf.ConfigProto()
config = tf.estimator.RunConfig(train_distribute=strategy, session_config=session_config,
                                log_step_count_steps=1, save_checkpoints_steps=float('inf'))
classifier = tf.estimator.Estimator(model_fn=model_fn, config=config)


x = np.array([1, 2, 3, 4])
y = np.array([5, 6, 7, 8])
train_input_fn = tf.estimator.inputs.numpy_input_fn(x, y, batch_size=1, num_epochs=None, shuffle=True)

tf.estimator.train_and_evaluate(
  classifier,
  train_spec=tf.estimator.TrainSpec(input_fn=lambda: train_input_fn, hooks=[RunHook()]),
  eval_spec=tf.estimator.EvalSpec(input_fn=lambda: train_input_fn)
)