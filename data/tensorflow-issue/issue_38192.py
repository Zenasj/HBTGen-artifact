import attr
from collections import namedtuple
import tensorflow as tf


@attr.s(frozen=True)
class DataContainer:
    data = attr.ib()
    
NamedDataContainer = namedtuple('NamedDataContainer', ['data'])
        
@tf.function
def f(x):
    return x

x = tf.constant(0.)
f(DataContainer(x)) # legit
f(NamedDataContainer(x)) # legit

dataset = tf.data.Dataset.from_tensor_slices([0., 1., 2., 3.])

dataset.map(NamedDataContainer) # legit
dataset.map(DataContainer) # not legit