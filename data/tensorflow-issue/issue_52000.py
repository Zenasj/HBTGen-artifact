import attr
import tensorflow as tf
from tensorflow.python.util import nest

@attr.attrs(auto_attribs=True)
class Container:
    a: object
    b: object

shape_object = Container(a=[1, 2], b=[3])
shallow_object = Container(a=None, b=None)
shape_res = nest.map_structure_up_to(shallow_object, tf.TensorShape, shape_object)