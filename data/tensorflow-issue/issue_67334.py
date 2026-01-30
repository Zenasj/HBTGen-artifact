import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Taken from the guide for reference (https://www.tensorflow.org/guide/extension_type#keras):

class Network(tf.experimental.BatchableExtensionType):
  shape: tf.TensorShape  # batch shape. A single network has shape=[].
  work: tf.Tensor        # work[*shape, n] = work left to do at node n
  bandwidth: tf.Tensor   # bandwidth[*shape, n1, n2] = bandwidth from n1->n2

  def __init__(self, work, bandwidth):
    self.work = tf.convert_to_tensor(work)
    self.bandwidth = tf.convert_to_tensor(bandwidth)
    work_batch_shape = self.work.shape[:-1]
    bandwidth_batch_shape = self.bandwidth.shape[:-2]
    self.shape = work_batch_shape.merge_with(bandwidth_batch_shape)

  def __repr__(self):
    return network_repr(self)


input_spec = Network.Spec(shape=None,
                          work=tf.TensorSpec(None, tf.float32),
                          bandwidth=tf.TensorSpec(None, tf.float32))
model = tf.keras.Sequential([
    tf.keras.layers.Input(type_spec=input_spec),
    # BalanceNetworkLayer(),
    ])