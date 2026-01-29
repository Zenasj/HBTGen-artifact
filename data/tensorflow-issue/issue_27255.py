# tf.random.uniform((B, 784), dtype=tf.float32) ← input shape [batch_size, 784]

import tensorflow as tf

def expand_bias_to(bias, shape):
    # utility function to reshape bias for broadcasting
    return tf.reshape(bias, [1] * (len(shape) - len(bias.shape)) + bias.shape.as_list())

class Function(object):
    def __init__(self, name=None):
        self._name = name

    def call(self, params, inputs, **kwargs):
        raise NotImplementedError("call not implemented for {}.".format(self.__dict__))

    def __call__(self, params, inputs, return_inputs=False, return_outputs=False, **kwargs):
        return self.call(params, inputs,
                         return_inputs=return_inputs,
                         return_outputs=return_outputs,
                         **kwargs)


class ParametricFunction(Function):
    def __init__(self, initializers=None, name=None):
        super(ParametricFunction, self).__init__(name=name)
        self.initializers = initializers

    def __call__(self, params, inputs, return_inputs=False, return_outputs=False, **kwargs):
        # The base __call__ calls the abstract call method.
        # Return (result, extra) tuple for interface compatibility.
        result, extra = super(ParametricFunction, self). \
            __call__(params, inputs,
                     return_inputs=return_inputs,
                     return_outputs=return_outputs,
                     **kwargs)
        return result, extra

    def call(self, params, inputs, **kwargs):
        # Must be implemented in subclasses
        raise NotImplementedError()


class Affine(ParametricFunction):
    def __init__(self, out_dim, name=None):
        super(Affine, self).__init__(initializers=None, name=name)
        self.out_dim = out_dim

    def call(self, params, inputs, **kwargs):
        # params expected to be tuple/list:
        # if length 2: (weight, bias)
        # else: (weight,), bias assumed zero
        if len(params) == 2:
            w, b = params
            b = expand_bias_to(b, inputs.shape)
        else:
            w, b = params[0], 0
        return tf.matmul(inputs, w) + b, None


class MyModel(tf.keras.Model):
    """
    This model encapsulates the Affine function as a submodule.
    Inputs: Tensor of shape [batch_size, 784]
    Params: tuple of (weight tensor [784,784], bias tensor [784])
    The forward returns logits as tf.matmul(input, W) + b
    """
    def __init__(self):
        super().__init__()
        self.affine = Affine(out_dim=784)

    def call(self, inputs_tuple, training=False):
        # inputs_tuple expected as (params, inputs)
        params, inputs = inputs_tuple
        # params is a tuple of tf.Variable or tf.Tensor: (W, b)
        logits, extra = self.affine(params, inputs, return_inputs=False, return_outputs=False)
        return logits


def my_model_function():
    # Create MyModel instance with no special weights initialization needed here
    return MyModel()


def GetInput():
    # Return a tuple (params, inputs)
    # params is tuple of weights and bias tensors compatible with Affine layer
    batch_size = 100  # reasonable batch size
    input_dim = 784
    output_dim = 784
    # Initialize random weights and bias tensors
    weights = tf.random.normal([input_dim, output_dim], dtype=tf.float32)
    bias = tf.zeros([output_dim], dtype=tf.float32)
    inputs = tf.random.uniform([batch_size, input_dim], dtype=tf.float32)
    params = (weights, bias)
    return (params, inputs)

