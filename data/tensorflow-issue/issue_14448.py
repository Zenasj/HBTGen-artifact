# tf.random.uniform((B,), dtype=tf.int64) ← input is an integer scalar index for from_indexable Dataset range map

import tensorflow as tf
from tensorflow.python.data.util import nest
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import script_ops
import traceback


class MyModel(tf.keras.Model):
    def __init__(self, iterator):
        """
        A Model that wraps an indexable python object (iterator) accessed via a
        py_func with optional parallel mapping.
        
        This fuses the notion of from_indexable Dataset, where an indexable
        python object is wrapped in TensorFlow Dataset `range` and a `map` with
        py_func to call into python code.
        
        Here, for demonstration, the model input is an integer index tensor,
        and the output is the python object element accessed by iterator[index].
        
        Args:
            iterator: Python indexable object (with __getitem__ and __len__)
        """
        super().__init__()
        self.iterator = iterator

    @tf.function(jit_compile=True)
    def call(self, index_tensor):
        """
        Forward pass for an index or a batch of indices as int64 tensor.
        Uses `my_py_func` logic (wrapped here) to call python code that fetches
        iterator[index].

        Args:
            index_tensor: scalar or tensor of shape (B,) int64 indices.

        Returns:
            The elements accessed by iterator[index_tensor], structured as nested types.
        """
        # We use nested py_func logic with output_types and output_shapes inferred
        # dynamically. Since we don't know the output structure statically, assume
        # output_types and shapes by querying one element on graph build.

        # For tf.function graph mode we must use a py_func wrapper below:
        # But we can cache output_types and shapes from a single element.
        # Here we assume single-element input for shape/type inference during init,
        # then reuse output_types/output_shapes for calls.

        # NOTE: In real use cases, this may need caching or static known output.

        def index_to_entry(idx):
            return self.iterator[idx.numpy() if tf.executing_eagerly() else idx]

        # Define output_types and shapes by probing first element
        # This is done lazily first time call is made
        if not hasattr(self, '_cached_output_types'):
            # Use a dummy index 0 for inference
            example = self.iterator[0]
            output_types = nest.map_structure(
                lambda x: tf.as_dtype(x.dtype) if hasattr(x, 'dtype') else tf.as_dtype(type(x)),
                example)
            output_shapes = nest.map_structure(
                lambda x: x.shape if hasattr(x, 'shape') else tf.TensorShape([]),
                example)
            self._cached_output_types = output_types
            self._cached_output_shapes = output_shapes
        else:
            output_types = self._cached_output_types
            output_shapes = self._cached_output_shapes

        @py_func_decorator(output_types=output_types, output_shapes=output_shapes, stateful=True)
        def py_index_to_entry(idx_tensor):
            # idx_tensor is a scalar tf.Tensor, convert to python int
            idx = idx_tensor.numpy() if tf.executing_eagerly() else idx_tensor
            return index_to_entry(idx)

        # If input is scalar index, expand dims to batch of 1 for mapping
        if index_tensor.shape.rank == 0:
            index_tensor = tf.expand_dims(index_tensor, 0)

        # Map over batch indices with py_func wrapper
        # Using tf.map_fn to apply py_func element-wise (could use Dataset map outside Model)
        outputs = tf.map_fn(py_index_to_entry, index_tensor, dtype=output_types, fn_output_signature=output_types)

        # If input was scalar, squeeze output batch dim back out
        if outputs is not None and index_tensor.shape.rank == 1 and outputs is not None:
            if outputs is not None and hasattr(outputs, 'shape') and outputs.shape.rank == (len(index_tensor.shape) + 1):
                outputs = nest.map_structure(lambda t: tf.squeeze(t, axis=0), outputs)

        return outputs


def my_py_func(func, args=(), kwargs={}, output_types=None, output_shapes=None, stateful=True, name=None):
    """
    Extended py_func that supports nested output_types and output_shapes,
    and callable output_types/output_shapes to infer dynamically.

    Args:
      func: Python function to call.
      args: tuple/list of inputs.
      kwargs: dict of keyword args.
      output_types: nested structure of tf.dtypes or callable returning them.
      output_shapes: nested structure of shape or callable returning them.
      stateful: If True, py_func is stateful.
      name: Optional operation name.

    Returns:
      Tensor or nested structure of Tensors.
    """
    if isinstance(args, list):
        args = tuple(args)

    if callable(output_types):
        output_types = output_types(*args, **kwargs)
    if callable(output_shapes):
        output_shapes = output_shapes(*args, **kwargs)

    flat_output_types = nest.flatten(output_types)

    # Pack args and kwargs into a nested tuple to reconstruct inside python function
    args_tuple = (args, kwargs)
    flat_args = nest.flatten(args_tuple)

    def python_function_wrapper(*py_args):
        try:
            py_args_unpacked, py_kwargs_unpacked = nest.pack_sequence_as(args_tuple, py_args)
            ret = func(*py_args_unpacked, **py_kwargs_unpacked)
            nest.assert_shallow_structure(output_types, ret)
        except Exception:
            traceback.print_exc()
            raise
        return nest.flatten(ret)

    flat_values = script_ops.py_func(
        python_function_wrapper, flat_args, flat_output_types, stateful=stateful, name=name)

    if output_shapes is not None:
        output_shapes = nest.map_structure_up_to(
            output_types, tensor_shape.as_shape, output_shapes)
        flattened_shapes = nest.flatten(output_shapes)
        for ret_t, shape in zip(flat_values, flattened_shapes):
            ret_t.set_shape(shape)

    return nest.pack_sequence_as(output_types, flat_values)


def py_func_decorator(output_types=None, output_shapes=None, stateful=True, name=None):
    """
    Decorator factory that wraps a python function into a my_py_func call with
    given output_types and output_shapes (can be callable).

    Usage:
        @py_func_decorator(output_types=..., output_shapes=...)
        def fn(...):
            ...
    """
    def decorator(func):
        def call(*args, **kwargs):
            return my_py_func(
                func,
                args, kwargs,
                output_types=output_types,
                output_shapes=output_shapes,
                stateful=stateful,
                name=name)
        return call
    return decorator


def from_indexable(iterator, output_types, output_shapes=None, num_parallel_calls=None, stateful=True, name=None):
    """
    Create a tf.data.Dataset from an indexable python object that supports __len__ and __getitem__.

    Uses tf.data.Dataset.range with map + py_func to access iterator elements via indices.

    Args:
      iterator: Python object with __len__ and __getitem__
      output_types: nested tf.DType structure matching output of iterator[i]
      output_shapes: optional nested tf.TensorShape structure
      num_parallel_calls: number of parallel calls in Dataset.map (optional)
      stateful: if True, py_func is stateful
      name: name for the map

    Returns:
      tf.data.Dataset generating elements of iterator in order, supports parallel calls.
    """
    length = len(iterator)
    ds = tf.data.Dataset.range(length)

    @py_func_decorator(output_types=output_types, output_shapes=output_shapes, stateful=stateful, name=name)
    def index_to_element(i):
        return iterator[i]

    ds = ds.map(index_to_element, num_parallel_calls=num_parallel_calls)
    return ds


def my_model_function():
    """
    Return an instance of MyModel with an example indexable iterator.
    For demonstration, iterator is a simple list of numpy arrays (or dicts).

    In practice, this object can be any indexable python object with __getitem__ and __len__.

    The resulting MyModel expects integer index tensor input, and returns the element
    from the iterator.

    Here, example iterator is a list of dicts with tensors to mimic complex structure.
    """
    import numpy as np

    # Example iterator with dict outputs having numpy arrays
    class ExampleIterator:
        def __init__(self):
            self.data = [
                {'features': np.array([1.0, 2.0], dtype=np.float32), 'label': np.array(0, dtype=np.int32)},
                {'features': np.array([3.0, 4.0], dtype=np.float32), 'label': np.array(1, dtype=np.int32)},
                {'features': np.array([5.0, 6.0], dtype=np.float32), 'label': np.array(0, dtype=np.int32)},
            ]

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    iterator = ExampleIterator()
    model = MyModel(iterator)
    return model


def GetInput():
    """
    Return a sample input tensor compatible with MyModel input.

    Since MyModel input is index tensor(s) for the from_indexable dataset/model,
    generate a scalar index or batch of indices to access iterator elements.

    Example here: single scalar or batch tensor of indices with dtype tf.int64
    Within valid index range (0 to length-1).
    """
    # Use batch of 2 indices for example, dtype int64
    # Adjust batch size and indices to valid range depending on iterator length
    input_indices = tf.constant([0, 1], dtype=tf.int64)
    return input_indices

