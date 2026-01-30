from tensorflow import keras
from tensorflow.keras import optimizers

lr_fn = PiecewiseConstantDecay()
opt = SGD(lr_fn)
opt = WrapOpt(opt)

import tensorflow as tf
from tensorflow.keras import layers, optimizers, models
print(tf.__version__)
class OptimizerWrapper(optimizers.Optimizer):
  def __init__(self, optimizer, name=None, **kwargs):
    super(OptimizerWrapper, self).__init__(name, **kwargs)
    self._optimizer = optimizer

  def _create_slots(self, var_list):
    self._optimizer._create_slots(var_list)

  def _resource_apply_dense(self, grad, var):
    return self._optimizer._resource_apply_dense(grad, var)

  def _resource_apply_sparse(self, grad, var):
    return self._optimizer._resource_apply_sparse(grad, var)

  def get_config(self):
    return self._optimizer.get_config()


model = tf.keras.Sequential()
model.add(layers.Dense(8))
x = tf.constant(12., shape=(5, 1, 2, 4))
boundaries = [100000, 110000]
values = [1.0, 0.5, 0.1]
learning_rate_fn = optimizers.schedules.PiecewiseConstantDecay(
    boundaries, values)
#learning_rate_fn = optimizers.schedules.ExponentialDecay(
#    0.1, decay_steps=100000, decay_rate=0.96, staircase=True)
#learning_rate_fn = optimizers.schedules.PolynomialDecay(
#    0.1, 10000, 0.01, power=0.5)
opt = optimizers.SGD(learning_rate=learning_rate_fn, momentum=1.0)
opt = OptimizerWrapper(opt)

@tf.function
def train_step(x):
  with tf.GradientTape(persistent=True) as tape:
    y = model(x)
    loss = tf.reduce_mean(y)

  grads = tape.gradient(loss, model.variables)
  opt.apply_gradients(zip(grads, model.variables))
  return loss

for i in range(3):
  loss = train_step(x)
  print("Loss:", loss)

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))

class SimplePiecewiseConstantDecay(
    tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, boundaries, values, name=None):
    super(SimplePiecewiseConstantDecay, self).__init__()
    if len(boundaries) != len(values) - 1:
      raise ValueError(
          "The length of boundaries should be 1 less than the length of values"
      )
    self.boundaries = boundaries
    self.values = values
    self.name = name

  def __call__(self, step):
    pred_fn_pairs = []
    pred_fn_pairs.append((step <= self.boundaries[0], lambda: self.values[0]))
    pred_fn_pairs.append((step > self.boundaries[-1], lambda: self.values[-1]))
    for low, high, v in zip(self.boundaries[:-1], self.boundaries[1:],
                            self.values[1:-1]):
      # Need to bind v here; can do this with lambda v=v: ...
      pred = (step > low) & (step <= high)
      pred_fn_pairs.append((pred, lambda v=v: v))

    # The default isn't needed here because our conditions are mutually
    # exclusive and exhaustive, but tf.case requires it.
    default = lambda: self.values[0]
    return tf.case(pred_fn_pairs, default, exclusive=True)

  def get_config(self):
    return {"boundaries": self.boundaries,
            "values": self.values,
            "name": self.name}

class OptimizerWrapper(optimizers.Optimizer):

  ...

  def apply_gradients(self,
                      grads_and_vars,
                      name=None,
                      experimental_aggregate_gradients=True):
    self._optimizer.apply_gradients(grads_and_vars, name,
                                    experimental_aggregate_gradients)

def _resource_apply_dense(self, grad, var):
  if var is elegible:
    self._optimizer._resource_apply_dense(...)
    new_var = prune(var)
    return var.assign(new_var)
  else:
    return self._optimizer._resource_apply_dense(...)

class OptimizerWrapper(optimizers.Optimizer):
  def __init__(self, optimizer, name=None, **kwargs):
    super(OptimizerWrapper, self).__init__(name, **kwargs)
    self._optimizer = optimizer

  def _prepare(self, var_list):
    return self._optimizer._prepare(var_list)

  def _create_slots(self, var_list):
    self._optimizer._create_slots(var_list)

  def _resource_apply_dense(self, grad, var, apply_state):
    return self._optimizer._resource_apply_dense(grad, var, apply_state)

  def _resource_apply_sparse(self, grad, var):
    return self._optimizer._resource_apply_sparse(grad, var)

  def get_config(self):
    return self._optimizer.get_config()