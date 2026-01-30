import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers

with self.writer.as_default():
         tf.summary.trace_on(graph=False, profiler=True)

with self.writer.as_default():
         tf.summary.trace_export(step=step, name="model_profile", profiler_outdir="./test_profile")

from tensorflow.python.eager import context

context.context().mirroring_policy = context.MIRRORING_ALL

@tf.function
def train_with_strategy(strategy, model, optimizer, dataset):
  def train_step(x, y):
    return _train_step(model, optimizer, x, y)

  def update_state(state, x_and_y):
    per_replica_loss = strategy.experimental_run_v2(train_step, x_and_y)
    state['loss'] = strategy.reduce(
        tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)       
    state['step'] += 1
    return state

  initial_state = {'loss': np.nan, 'step': 0}
  return dataset.reduce(initial_state, update_state)

def create_distributed_optimizer(optimizer, name=None, device_dense='', device_sparse='',
                         compression=hvd.Compression.none, sparse_as_dense=False):
    class _DistributedOptimizer(tf.keras.optimizers.Optimizer):
        def __init__(self, name, device_dense, device_sparse, compression, sparse_as_dense, config):
            if name is None:
                name = "Distributed%s" % self.__class__.__base__.__name__
            self._allreduce_grads = hvd._make_allreduce_grads_fn(
                name, device_dense, device_sparse, compression, sparse_as_dense)
            super(self.__class__, self).__init__(**config)

        def apply_gradients(self, grads_and_vars, *args, **kwargs):
            if hvd.size() > 1:
                grads, vars = zip(*grads_and_vars)
                avg_grads = self._allreduce_grads(grads)
                grads_and_vars = list(zip(avg_grads, vars))
            return super(self.__class__, self).apply_gradients(grads_and_vars, *args, **kwargs)

        @classmethod
        def from_config(cls, cfg):
            return cls(name, device_dense, device_sparse, compression, sparse_as_dense, cfg)

    cls = type(optimizer.__class__.__name__, (optimizer.__class__,), dict(_DistributedOptimizer.__dict__))
    return cls(name, device_dense, device_sparse, compression, sparse_as_dense, optimizer.get_config())