class EagerFunc(object):
  """A wrapper for a function owned by an EagerPyFunc."""

  def __init__(self, func, Tout, is_grad_func):
    """Constructs an EagerFunc.
    Args:
      func: The function to wrap.
      Tout: A list of datatypes for the output; an empty list if the output is
        None.
      is_grad_func: Whether this EagerFunc is the gradient of another
        EagerPyFunc.
    """
    self._func = func
    self._out_dtypes = Tout
    self._is_grad_func = is_grad_func

    context.ensure_initialized()

  def _convert(self, value, dtype):
    """Converts `value` to a tensor of type `dtype`, with error checking.
    Args:
      value: The tensor to convert.
      dtype: The desired dtype.
    Returns:
      A tensor of type `dtype`, or a zeros tensor if value is None and
      this function is in fact a grdient function.
    Raises:
      RuntimeError: if `value` is a variable.
    """

    if isinstance(value, resource_variable_ops.ResourceVariable):
      raise RuntimeError(
          "Attempting to return a variable from an eagerly executed py_func. "
          "Only numeric data structures like Tensors or NumPy arrays should "
          "be returned; to return the value of a variable, make sure to obtain "
          "the Tensor backing it by calling `.read_value()` on the variable in "
          "question: %s" % value)
    if value is None and self._is_grad_func:
      # Gradient functions may legitimately return a list that contains
      # both Tensors and Python Nones. Unfortuantely this breaks the
      # OpKernel, so for now we replace None objects with zeros, which is
      # mathematically correct but will prevent short-circuiting gradient
      # computations.
      #
      # TODO(akshayka): Make it possible to return a list of both Tensors and
      # Nones from an EagerPyFunc.
      return constant_op.constant(0.0, dtype=dtype)
    return ops.convert_to_tensor(value, dtype=dtype)