import random

import tensorflow as tf

def foo(label, tensor=None, min_value=None, max_value=None):
    if tensor is not None and tf.is_tensor(tensor):
        result = tf.identity(tensor, name=label)
    else:
        result = tf.random.uniform(shape=(1, len(min_value)),
                                   minval=min_value,
                                   maxval=max_value, name=label)
    result._bar = [label]
    return result


@tf.function
def fooAutoGraph(label, tensor=None, min_value=None, max_value=None):
    if tensor is not None and tf.is_tensor(tensor):
        result = tf.identity(tensor, name=label)
    else:
        result = tf.random.uniform(shape=(1, len(min_value)),
                                   minval=min_value,
                                   maxval=max_value, name=label)
    result._bar = [label]
    return result


assert hasattr(foo('a', min_value=[0.] * 10, max_value=[1.] * 10), '_bar') # this works
assert hasattr(fooAutoGraph('a', min_value=[0.] * 10, max_value=[1.] * 10), '_bar') # this will fail

class MyLayer(KL.Layer):

    def __init__(self, *args, **kwargs):
        super(MyLayer, self).__init__(*args, **kwargs)

    def __call__(self, inputs, *args, **kwargs):
        outputs = super(MyLayer, self).__call__(inputs, *args, **kwargs)
        flat_outputs = nest.flatten(outputs)

        meta = self.compute_meta(inputs, **kwargs)
        for output in flat_outputs:
            output._my_meta = meta

        return outputs

    def compute_meta(self, inputs, **kwargs):
        return [] # no metadata