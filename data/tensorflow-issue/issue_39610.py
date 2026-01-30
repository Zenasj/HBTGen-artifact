import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def test_grads_non_zero(self):
    model = classification_models.TwoLayerLinalgConv(1, 784)
    with tf.GradientTape() as g:
      result = model(MNIST_DATA)
      loss = tf.keras.losses.sparse_categorical_crossentropy(MNIST_CLASSES, result, from_logits=True)
    grads = g.gradient(loss, model.trainable_weights)
    for idx, grad_elem in enumerate(grads):
      self.assertFalse(grad_elem is None)
      self.assertNotEqual(self.evaluate(tf.reduce_sum(tf.math.abs(grad_elem))), 0.)

class TwoLayerLinalgConv(tf.keras.Model):
  """Simple 2-dimensional convolution-like model via custom layer."""

  def __init__(
      self,
      n_filters,
      data_size,
      # ...other things
      ):
    super(TwoLayerLinalgConv, self).__init__()
    # Some initializations
    # etc
    self.conv1 = layers.LaplacianLayer(n_filters=self._n_filters,
                                       n_input_channels=self._n_input_channels,
                                       n_params_to_fit=self._kernel_size,
                                       n_dimensions=self._n_dimensions)
        # Must have 1 filter here for the dense layer in the end.
    self.conv2 = layers.LaplacianLayer(n_filters=1,
                                       n_input_channels=n_filters,
                                       n_params_to_fit=self._kernel_size,
                                       n_dimensions=self._n_dimensions)
    self.dense = tf.keras.layers.Dense(output_dim, activation=None)

  def call(self, inputs):
    x = tf.nn.relu(self.conv1(tf.reshape(inputs, [-1, self._n_input_channels, self._data_size])))
    x = tf.nn.relu(self.conv2(x))
    return self.dense(tf.reshape(x, [-1, self._data_size]))

def test_operator_trains(self):
    l = layers.LaplacianLayer(1, 1, 1)
    l.build([10, 1, MNIST_SIZE])
    with tf.GradientTape() as g:
      input_with_channel = tf.reshape(MNIST_DATA, [-1, 1, MNIST_SIZE])
      result_of_layer = l(input_with_channel)
      pred = tf.keras.layers.Dense(10)(result_of_layer)
      result = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
          MNIST_LABELS, tf.reshape(pred, shape=[10, 10]), from_logits=True))

    grad = g.gradient(result, l.trainable_weights)
    for grad_elem in grad:
      self.assertFalse(grad_elem is None)
      self.assertNotEqual(self.evaluate(tf.reduce_sum(tf.math.abs(grad_elem))), 0.)