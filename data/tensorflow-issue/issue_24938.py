import tensorflow as tf
from tensorflow import keras

class ReshapeLayer(Layer):
    def __init__(self, **kwargs):
        super(ReshapeLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        super(ReshapeLayer, self).build(input_shape)

    def call(self, x):
        s = K.shape(x)
        # zeros_w = K.zeros((s[0], 1, s[2], s[3]), tf.float32) # does not work
        zeros_w = tf.zeros((s[0], 1, s[2], s[3]), tf.float32)
        r = K.concatenate([x, zeros_w], 1)

        s = K.shape(r)
       #  zeros_h = K.zeros((s[0], s[1], 1, s[3]), tf.float32)  # does not work
        zeros_h = tf.zeros((s[0], s[1], 1, s[3]), tf.float32)
        r = K.concatenate([r, zeros_h], 2)
        return r

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[1] = shape[1] + 1
        shape[2] = shape[2] + 1
        return tf.TensorShape(shape)

# code of track_variable in tf.keras.backend.py
def track_variable(v):
  """Tracks the given variable for initialization."""
  if context.executing_eagerly():
    return
  graph = v.graph if hasattr(v, 'graph') else ops.get_default_graph()
  if graph not in _GRAPH_VARIABLES:
    _GRAPH_VARIABLES[graph] = weakref.WeakSet()
  _GRAPH_VARIABLES[graph].add(v)