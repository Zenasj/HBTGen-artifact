import tensorflow as tf
from tensorflow import keras

class MyModel(keras.Model):
    def __init__(self, **kwargs):
        super(MyModel, self).__init__(**kwargs)
    def call(self, inputs):
        self.add_loss(tf.reduce_mean(tf.square(inputs)))
        return inputs

x = tf.placeholder(tf.float32, [None, 5])
model = MyModel()
model(x)

model.losses  # returns []
model._losses # returns [<tf.Tensor 'my_model/Mean:0' shape=() dtype=float32>]

list(set(relevant_conditional_losses + unconditional_losses + self._losses))

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.eager import context
from tensorflow.python.keras.utils import tf_utils


class MyModel(keras.Model):

  def __init__(self, **kwargs):
    super(MyModel, self).__init__(**kwargs)

  def call(self, inputs):
    self.add_loss(tf.reduce_mean(tf.square(inputs)))
    return inputs


x = tf.placeholder(tf.float32, [None, 5])
model = MyModel()
model(x)

print('model.losses: ', model.losses)  # returns []
print('model._losses: ', model._losses)
# returns [<tf.Tensor 'my_model/Mean:0' shape=() dtype=float32>]


def losses(self):
  # this function is copied from `Network.losses`, but revised the line commented.
  losses = self._unfiltered_losses
  if context.executing_eagerly():
    return losses
  relevant_inputs = []
  for i in range(0, len(self._inbound_nodes)):
    inputs = self.get_input_at(i)
    if isinstance(inputs, list):
      relevant_inputs += inputs
    else:
      relevant_inputs.append(inputs)
  ########## revised here!  ##########
  if not relevant_inputs:  
    # return losses
    return list(set(losses + self._losses))
  ###############################
  reachable = tf_utils.get_reachable_from_inputs(relevant_inputs, losses)
  relevant_conditional_losses = [x for x in losses if x in reachable]
  unconditional_losses = [x for x in losses if x._unconditional_loss]  # pylint: disable=protec
  return list(
    set(relevant_conditional_losses + unconditional_losses + self._losses))


print(losses(model))