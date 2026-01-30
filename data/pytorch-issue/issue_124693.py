import torch

@register_decomposition(aten.masked_fill)
@out_wrapper()
def masked_fill(a: TensorLikeType, mask: TensorLikeType, value: TensorOrNumberLikeType):
    python_type = utils.dtype_to_type(a.dtype)
    if isinstance(value, Number):
        value_type = type(value)
    else:
        # NOTE: Could not use value = item(value) as it resulted in
        # RuntimeError: Cannot cast FakeTensor(cpu) to number
        value_ndim = value.ndim
        torch._check(
            value_ndim == 0,
            lambda: f"only supports a 0-dimensional value tensor, but got tensor with {value_ndim} dimension",
        )
        # `masked_fill` allows cpu scalar to be moved to cuda and xpu but not otherwise.
        is_cpu_scalar = a.device.type in ["cuda", "xpu"] and value.device.type == "cpu"
        torch._check(
            is_cpu_scalar or value.device == a.device,
            lambda: "Expected `value` to be on same device as `a`",
        )
        value_type = utils.dtype_to_type(value.dtype)