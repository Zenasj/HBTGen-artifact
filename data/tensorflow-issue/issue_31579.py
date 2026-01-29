# tf.random.normal((B, 2, 3), dtype=tf.float64) ← example input shape based on MNISTWidth=2, MNISTHeight=3 with batch dimension None

import tensorflow as tf
import inspect

from typing import Generic, Any, TypeVar

# Type variables for generic shape and data type
ShapeType = TypeVar("ShapeType")
DataType = TypeVar("DataType")

# Define generic 3D shape container for type annotations
class Shape3D(Generic[ShapeType, ShapeType, ShapeType]):
    pass

# Dimension base class and specializations
class Dimension(object):
    value = NotImplemented

class Dynamic(Dimension):
    value = None  # Represents unknown dimension size (None in tf.TensorSpec)

class Float32(Dimension):
    value = tf.float32

class Float64(Dimension):
    value = tf.float64

# Generic Tensor annotation class carrying shape and dtype info for input_signature construction
class Tensor(Generic[ShapeType, DataType]):
    def __rmul__(self, other: Any):
        pass  # Place-holder for mypy; no actual runtime computation here

def function(fn):
    """Decorator that inspects type annotations of fn arguments and constructs tf.function with input_signature.

    Expects arguments annotated as Tensor[Shape3D[Dim1, Dim2, Dim3], Dtype]. Supports only positional args.
    """
    argspec = inspect.getfullargspec(fn)
    if argspec.varargs is not None or argspec.varkw is not None:
        # For simplicity only support functions with fixed positional arguments
        raise NotImplementedError("Only supports positional arguments without *args or **kwargs")

    input_signature = []
    for name in argspec.args:
        # Get annotation type for argument
        ann = argspec.annotations.get(name, None)
        if ann is None:
            raise ValueError(f"Argument '{name}' must be annotated with Tensor type")

        # Extract the shape and dtype type arguments from the annotation
        # ann should be Tensor[Shape3D[...], Dtype]
        try:
            shape_as_type, dtype = ann.__args__
        except AttributeError:
            raise TypeError(f"Argument '{name}' annotation must be Tensor with shape and dtype parameters")

        # Parse the shape dimensions from the shape_as_type which is Shape3D[Dim1, Dim2, Dim3]
        shape_dims = []
        try:
            for dim in shape_as_type.__args__:
                if not hasattr(dim, "value"):
                    raise ValueError(f"Dimension {dim} must have a 'value' attribute")
                val = dim.value
                if val is None:
                    shape_dims.append(None)
                else:
                    shape_dims.append(int(val))
        except Exception as e:
            raise ValueError(f"Failed to parse shape dimensions for argument '{name}': {e}")

        # Get the TF dtype from dtype.value
        if not hasattr(dtype, "value"):
            raise ValueError(f"Dtype {dtype} must have a 'value' attribute")
        tf_dtype = dtype.value

        ts = tf.TensorSpec(shape=shape_dims, dtype=tf_dtype, name=name)
        input_signature.append(ts)

    # Return a tf.function with the constructed input_signature and autograph=True by default
    return tf.function(fn, input_signature=tuple(input_signature))


# === User code defining dimensionality ===

class MNISTWidth(Dimension):
    value = 2

class MNISTHeight(Dimension):
    value = 3


# === User's Keras Model using the above function decorator and annotations ===

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Simple example model with one Conv2D layer adapted to input shape (..., 2, 3)
        # Assume input shape is (batch, None, 2, 3) from annotations — batch is dynamic, 2x3 fixed spatial dims

        # For simplicity, just flatten the last dims and use Dense layer:
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(10, activation='relu')
        self.output_layer = tf.keras.layers.Dense(1)  # regression or binary/logit output

    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense(x)
        return self.output_layer(x)


@function
def my_model_function(
    x: Tensor[Shape3D[Dynamic, MNISTWidth, MNISTHeight], Float64]
):
    # Instantiate model lazily
    model = MyModel()
    return model(x)


def GetInput():
    # Return a random input tensor matching MyModel input spec: (batch, 2, 3), dtype float64
    # Batch size chosen arbitrarily as 5 here for demo
    return tf.random.uniform((5, 2, 3), dtype=tf.float64)

