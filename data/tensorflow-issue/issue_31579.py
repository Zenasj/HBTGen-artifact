import random

python
import tensorflow as tf


def function(fn):
    input_signature = list(fn.__annotations__.values())
    return tf.function(fn, autograph=False, input_signature=input_signature)


@function
def foo(
    x: tf.TensorSpec(shape=[None], dtype=tf.float64),
    y: tf.TensorSpec(shape=[None], dtype=tf.float64),
):
    return x + 10.0 + y


vec32 = tf.random.normal([2], dtype=tf.float32)
vec64 = tf.random.normal([2], dtype=tf.float64)


# should pass
foo(vec64, vec64)
foo(y=vec64, x=vec64)

# should fail
foo(vec32, vec64)

python
@tf.function(
    autograph=False,
    input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.float64),
        tf.TensorSpec(shape=[None], dtype=tf.float64),
    ],
)
def foo(x, y):
    return x + 10.0 + y

python
import tensorflow as tf

from typing import Generic, TypeVar
from typing_extensions import Literal


ShapeType = TypeVar("ShapeType")
DataType = TypeVar("DataType")

Shape = Literal


class Float32:
    dtype = tf.float32


class Float64:
    dtype = tf.float64


# TODO(jeff): generate all dtypes


class Tensor(Generic[ShapeType, DataType]):
    @classmethod
    def shape(self):
        return self.__args__[0].__values__

    @classmethod
    def dtype(self):
        return self.__args__[1].dtype

    def __add__(self, other):
        return self + other


def function(fn):
    annotation_values = fn.__annotations__.values()
    tensor_specs = [tf.TensorSpec(x.shape(), x.dtype()) for x in annotation_values]
    return tf.function(fn, input_signature=tensor_specs)


@function
def foo(x: Tensor[Shape[None, 2, 3], Float64]):
    return x + 42.0


foo(tf.random.normal([1, 2, 3], dtype=tf.float64))  # OK
foo(tf.random.normal([2, 2, 3], dtype=tf.float64))  # OK
foo(tf.random.normal([1, 2, 3, 4], dtype=tf.float64))  # NOT OK
foo(tf.random.normal([1, 2, 3], dtype=tf.float32))  # NOT OK

import tensorflow as tf

import inspect
import typing
from typing import Any, Generic, TypeVar, get_type_hints
from typing import NewType
from typing_extensions import Literal

ShapeType = TypeVar("ShapeType")
DataType = TypeVar("DataType")


class Shape(Generic[ShapeType]):
  pass


class Float32(object):
    value = tf.float32


class Float64(object):
    value = tf.float64


# TODO(jeff): generate all dtypes


class Tensor(Generic[ShapeType, DataType]):
  def __rmul__(self, other: Any):
    pass  # Just appeasing mypy here, the real Tensor has a proper implementation.

  pass


def function(fn):
    argspec = inspect.getfullargspec(fn)
    if (argspec.varargs is not None or argspec.varkw is not None or argspec.varkw is not None):
      raise NotImplemented('only positional args for now')

    input_signature = []
    for name in argspec.args:
      if name not in argspec.annotations:
        input_signature.append(None)
        continue
      shape_as_type, dtype = argspec.annotations[name].__args__
      shape = []
      for s in shape_as_type.__args__[0].__values__:
        if s is None:
          shape.append(None)
        else:
          shape.append(int(s))

      ts = tf.TensorSpec(shape=shape, dtype=dtype.value)
      input_signature.append(ts)
    return tf.function(fn, input_signature=input_signature)


@function
def foo(x: Tensor[Shape[Literal[(None, 2, 3)]], Float64]):
    return 2 * x

foo(tf.random.normal([1, 2, 3], dtype=tf.float64))  # OK
foo(tf.random.normal([2, 2, 3], dtype=tf.float64))  # OK
try:
  foo(tf.random.normal([1, 2, 3, 4], dtype=tf.float64))  # NOT OK
  assert False
except ValueError:
  pass
try:
  foo(tf.random.normal([1, 2, 3], dtype=tf.float32))  # NOT OK
  assert False
except ValueError:
  pass

## This is what the gigantic file of type defs would contain

Shape3DDim1 = TypeVar("Shape3DDim1")
Shape3DDim2 = TypeVar("Shape3DDim2")
Shape3DDim3 = TypeVar("Shape3DDim3")

class Shape3D(Generic[Shape3DDim1, Shape3DDim2, Shape3DDim3]):
  pass

class Dimension(object):
  value = NotImplemented

class Dynamic(Dimension):
  value = None

## This is what the user would have to define:

class MNISTWidth(Dimension):
  value = 2

class MNISTHeight(Dimension):
  value = 3


@function
def foo(x: Tensor[Shape3D[Dynamic, MNISTWidth, MNISTHeight], Float64]):
    return 2 * x

python
@tf.function(input_signature=[None, tf.TensorSpec([1, 2], tf.float32)])
def foo(x, y):
    return x + y

python
import tensorflow as tf
import inspect

from typing import Generic, Any, TypeVar

# TODO: generate all dtypes
# TODO: generate all shapes

ShapeType = TypeVar("ShapeType")
DataType = TypeVar("DataType")

Shape3DDim1 = TypeVar("Shape3DDim1")
Shape3DDim2 = TypeVar("Shape3DDim2")
Shape3DDim3 = TypeVar("Shape3DDim3")


class Shape3D(Generic[Shape3DDim1, Shape3DDim2, Shape3DDim3]):
    pass


class Dimension(object):
    value = NotImplemented


class Dynamic(Dimension):
    value = None


class Float32(object):
    value = tf.float32


class Float64(object):
    value = tf.float64


class Tensor(Generic[ShapeType, DataType]):
    def __rmul__(self, other: Any):
        pass  # Just appeasing mypy here, the real Tensor has a proper implementation.


def function(fn):
    argspec = inspect.getfullargspec(fn)
    if argspec.varargs is not None or argspec.varkw is not None:
        raise NotImplemented("only positional args for now")

    input_signature = []
    for name in argspec.args:
        shape_as_type, dtype = argspec.annotations[name].__args__
        shape = []
        for s in shape_as_type.__args__:
            if s.value is None:
                shape.append(None)
            else:
                shape.append(int(s.value))

        ts = tf.TensorSpec(shape=shape, dtype=dtype.value, name=name)
        input_signature.append(ts)
    return tf.function(fn, input_signature=input_signature)


# User code starts here
class MNISTWidth(Dimension):
    value = 2


class MNISTHeight(Dimension):
    value = 3


@function
def foo(x: Tensor[Shape3D[Dynamic, MNISTWidth, MNISTHeight], Float64]):
    return 2.0 * x


# Some ad hoc testing
print(f"foo signature: {foo.input_signature}")
foo_x_ts = tf.TensorSpec(shape=[None, 2, 3], dtype=tf.float64, name="x")
assert len(foo.input_signature) == 1
assert foo.input_signature[0] == foo_x_ts


@function
def bar():
    return tf.random.normal([1, 2, 3])


print(f"bar signature: {bar.input_signature}")
assert bar.input_signature == ()