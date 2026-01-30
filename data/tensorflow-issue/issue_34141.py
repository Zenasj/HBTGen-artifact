from tensorflow import keras
from tensorflow.keras import layers

import tensorflow as tf

class MyLayer(tf.keras.layers.Layer):
    def build(self, input_shape):

        if input_shape is None:
            print('error: input shape is none')
        else:
            print('build:', input_shape)

        super().build(input_shape)

    def call(self, inputs):
        print('call:', inputs.shape)

        return inputs * 2

def main():

    l = MyLayer()

    inp = tf.keras.Input(shape=(5, 5, 1))
    l.compute_output_shape((None, 5, 5, 1))

    o = l(inp)

    print("input_shape:", l.input_shape)
    print("count_params:", l.count_params())
    print('output:', o.shape)

if __name__=='__main__':
    main()

def _maybe_build_with_shapes(self, inputshapes):
    # Check input assumptions set before layer building, e.g. input rank.
    if not self.built:

      # Only call `build` if the user has manually overridden the build method.
      if not hasattr(self.build, '_is_default'):
        # Any setup work performed only once should happen in an `init_scope`
        # to avoid creating symbolic Tensors that will later pollute any eager
        # operations.
        with tf_utils.maybe_init_scope(self):
          self.build(input_shapes)
      # We must set self.built since user defined build functions are not
      # constrained to set self.built.
      self.built = True

def compute_output_shape(self, input_shape):
    """Computes the output shape of the layer.
    If the layer has not been built, this method will call `build` on the
    layer. This assumes that the layer will later be used with inputs that
    match the input shape provided here.
    Arguments:
        input_shape: Shape tuple (tuple of integers)
            or list of shape tuples (one per output tensor of the layer).
            Shape tuples can include None for free dimensions,
            instead of an integer.
    Returns:
        An input shape tuple.
    """
    if context.executing_eagerly():
      # In this case we build the model first in order to do shape inference.
      # This is acceptable because the framework only calls
      # `compute_output_shape` on shape values that the layer would later be
      # built for. It would however cause issues in case a user attempts to
      # use `compute_output_shape` manually with shapes that are incompatible
      # with the shape the Layer will be called on (these users will have to
      # implement `compute_output_shape` themselves).
      with context.graph_mode():
        graph = func_graph.FuncGraph('graph')
        with graph.as_default():
          input_shape = tf_utils.convert_shapes(input_shape, to_tuples=False)
          inputs = nest.map_structure(
              base_layer_utils.generate_placeholders_from_shape, input_shape)
          self._maybe_build(inputs)
          try:
            if self._expects_training_arg:
              outputs = self(inputs, training=False)
            else:
              outputs = self(inputs)
          except TypeError:
            raise NotImplementedError('We could not automatically infer '
                                      'the static shape of the layer\'s output.'
                                      ' Please implement the '
                                      '`compute_output_shape` method on your '
                                      'layer (%s).' % self.__class__.__name__)
      return nest.map_structure(lambda t: t.shape, outputs)
    raise NotImplementedError