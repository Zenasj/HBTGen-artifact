# torch.rand(2, 3, 5, 5, dtype=torch.float16)
import torch
from torch import nn
from typing import Tuple
import functools

def _custom_setup_context(
    setup_context_fn=None,
    *,
    device_type: str,
    cast_inputs: torch.dtype = None,
):
    """The missing amp setup_context decorator for custom ops."""
    if setup_context_fn is None:
        return functools.partial(
            _custom_setup_context, device_type=device_type, cast_inputs=cast_inputs
        )

    @functools.wraps(setup_context_fn)
    def decorate_setup_context(ctx, *args, **kwargs):
        ctx._dtype = torch.get_autocast_dtype(device_type)
        if cast_inputs is None:
            ctx._fwd_used_autocast = torch.is_autocast_enabled(device_type)
            return setup_context_fn(ctx, *args, **kwargs)
        else:
            autocast_context = torch.is_autocast_enabled(device_type)
            ctx._fwd_used_autocast = False
            if autocast_context:
                with torch.autocast(device_type=device_type, enabled=False):
                    args_cast = torch.amp.autocast_mode._cast(
                        args, device_type, cast_inputs
                    )
                    kwargs_cast = torch.amp.autocast_mode._cast(
                        kwargs, device_type, cast_inputs
                    )
                    return setup_context_fn(ctx, *args_cast, **kwargs_cast)
            else:
                return setup_context_fn(ctx, *args, **kwargs)

    return decorate_setup_context

@torch.library.custom_op("bar::foo", mutates_args=())
@torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float16)
def foo(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
) -> torch.Tensor:
    assert input.dtype == torch.float16
    assert weight.dtype == torch.float16
    assert bias is None or bias.dtype == torch.float16
    # Dummy computation for testing
    return input + weight.sum() + (bias.sum() if bias is not None else 0)

@foo.register_fake
def _(input, weight, bias=None):
    # Fake implementation for testing
    return input + weight.sum() + (bias.sum() if bias is not None else 0)

@torch.library.custom_op("bar::foo_backward", mutates_args=())
def foo_backward_op(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    grad_output: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert grad_output.dtype == torch.float16
    assert input.dtype == torch.float16
    assert weight.dtype == torch.float16
    # Dummy backward implementation
    return (
        torch.zeros_like(input),
        torch.zeros_like(weight),
        torch.zeros_like(bias),
    )

@foo_backward_op.register_fake
def _(input, weight, bias, grad_output):
    return (
        torch.zeros_like(input),
        torch.zeros_like(weight),
        torch.zeros_like(bias),
    )

@torch.amp.custom_bwd(device_type="cuda")
def foo_backward(ctx, grad_output):
    input, weight, bias = ctx.saved_tensors
    grad_input, grad_weight, grad_bias = foo_backward_op(
        input, weight, bias, grad_output
    )
    return grad_input, grad_weight, grad_bias

@_custom_setup_context(device_type="cuda", cast_inputs=torch.float16)
def foo_setup_context(ctx, inputs, output):
    input, weight, bias, *_ = inputs
    ctx.save_for_backward(input, weight, bias)

foo.register_autograd(foo_backward, setup_context=foo_setup_context)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(64, 3, 3, 3, dtype=torch.float16))
        self.bias = nn.Parameter(torch.randn(64, dtype=torch.float16))

    def forward(self, x):
        return foo(x, self.weight, self.bias)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 3, 5, 5, dtype=torch.float16)

