import random

from abc import abstractmethod
import tensorflow as tf
import logging


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


def expand_bias_to(bias, shape):
    return tf.reshape(bias, [1] * (len(shape) - len(bias.shape)) + bias.shape.as_list())


def call_or_get(maybe_callable, *args, **kwargs):
    if callable(maybe_callable):
        return maybe_callable(*args, **kwargs)
    else:
        return maybe_callable


class ParametricFunction(Function):
    def __init__(self, initializers, name=None):
        super(ParametricFunction, self).__init__(name=name)
        self.initializers = initializers

    def __call__(self, params, inputs, return_inputs=False, return_outputs=False, **kwargs):
        result, extra = super(ParametricFunction, self). \
            __call__(params, inputs,
                     return_inputs=return_inputs,
                     return_outputs=return_outputs,
                     **kwargs)
        return result, extra

    @abstractmethod
    def call(self, params, inputs, **kwargs):
        raise NotImplementedError()


class Affine(ParametricFunction):
    def __init__(self, out_dim,
                 name=None):
        super(Affine, self).__init__(initializers=None, name=name)
        self.out_dim = out_dim

    def call(self, params, inputs, **kwargs):
        if len(params) == 2:
            w, b = params
            b = expand_bias_to(b, inputs.shape)
        else:
            w, b = params[0], 0
        return tf.matmul(inputs, w) + b, None


# !!! IF THIS IS COMMENTED OUT IT WORKS !!!
@tf.function
def train_one_step(variables, batch):
    model = Affine(784)
    logits, extra = model(variables, batch, return_inputs=False, return_outputs=False)
    loss_value = tf.reduce_mean(logits ** 2)
    return loss_value


def main():
    batch_size = 100
    tf.get_logger().setLevel(logging.WARNING)
    params = (tf.Variable(initial_value=tf.random.normal([784, 784]), name="W"),
              tf.Variable(initial_value=tf.zeros([784]), name="b"))
    data = tf.random.normal([batch_size, 784])

    for epoch in range(10):
        loss = train_one_step(params, data)
        epoch += 1
        print("Epoch {}, loss: {:0.3f}".format(epoch, loss))


if __name__ == '__main__':
    main()