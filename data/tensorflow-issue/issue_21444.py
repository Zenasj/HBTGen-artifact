import tensorflow as tf

eval_spec = tf.estimator.EvalSpec(
        input_fn=eval_input_fn,
        steps=100,
        start_delay_secs=0,
        throttle_secs=5,
        hooks=[MyEvalHook()]
    )

from tensorflow_core.python.training.session_run_hook import SessionRunArgs, SessionRunHook

class MyEvalHook(SessionRunHook):

    def __init__(self):
        pass

    def before_run(self, run_context):
        return SessionRunArgs(
            dict(
                global_step=tf.compat.v1.train.get_or_create_global_step()
            )
        )

    def after_run(self, run_context, run_values):
        pass