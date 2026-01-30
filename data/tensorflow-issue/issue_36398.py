import tensorflow as tf

class LossScaleOptimizer(Optimizer):
  def __init__(self, optimizer):
    super(LossScaleOptimizer, self).__init__()
    self._optimizer = optimizer

  def _process_loss(self, loss):
    return self._get_scaled_loss(loss)

  def _process_unaggregated_gradients(self, gradients):
    return self._get_unscaled_gradients(gradients)

optimizer = LossScaleOptimizer(tf.optimizers.Adam(1e-3))
with tf.GradientTape() as tape:
  loss = ...  # No scaling or other special-casing of LossScaleOptimizer needed here.
  optimizer.minimize(loss, variables, tape=tape)

class CombinedOptimizer(Optimizer):
  def __init__(self, optimizer1, optimizer2):
    """Calls hooks of optimizer1, then optimizer2."""
    super(CombinedOptimizer, self).__init__()
    self._optimizer1 = optimizer1
    self._optimizer2 = optimizer2

  def _process_loss(self, loss):
    loss = self._optimizer1._process_loss(loss)
    return self._optimizer2._process_loss(loss)

  # Same for other hooks.
  ...

def train_step(data):
  x, y, sw = unpack(data)
  with tf.GradientTape() as tape:
    y_pred = model(x, training=True)
    loss = model.compiled_loss(y, y_pred, sw)
    loss += tf.add_n(model.losses)
  model.optimizer.minimize(loss, model.trainable_variables, tape=tape)

with tf.GradientTape() as tape:
    loss = loss_fn(features, labels)
    scaled_loss = optimizer.get_scaled_loss(loss)
scaled_grads = tape.gradient(loss, model.trainable_variables)
# apply_gradients will unscale gradients, but not scale loss
optimizer.apply_gradients(list(zip(fp32_scaled_grads, 
                                   model.trainable_variables)))

# Pseudo-code to illustrate
opt = SGD(...)
opt = HVDDistributedOpt(opt, ...)
opt = LossScaleOpt(opt, ...)

# Configuration A
opt = SGD(...)
opt = HVDDistributedOpt(opt, ...)
opt = LossScaleOpt(opt, ...)

# Configuration B
opt = SGD(...)
opt = LossScaleOpt(opt, ...)
opt = HVDDistributedOpt(opt, ...)