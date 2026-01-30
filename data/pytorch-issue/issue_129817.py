import torch.nn as nn

import logging
from typing import List

import torch
from functorch.compile import aot_module_simplified, make_boxed_func


@torch.library.custom_op("mylib::somefunc_forward", mutates_args=())
def somefunc_forward(
    input_: torch.Tensor,
    weight: torch.Tensor,
    shape: List[int],
) -> torch.Tensor:
    return torch.ones_like(input_)


@somefunc_forward.register_fake
def _(input_, shape, weight):
    return torch.empty_like(input_)


@torch.library.custom_op("mylib::somefunc_backward", mutates_args=())
def somefunc_backward(
    grad_output: torch.Tensor,
    input_: torch.Tensor,
    weight: torch.Tensor,
    shape: List[int],
) -> torch.Tensor:
    print(f"backward.{grad_output.shape=}")
    print(f"backward.{input_.shape=}")
    print(f"backward.{weight.shape=}")
    print(f"backward.{shape=}")
    assert list(weight.shape) == shape
    return torch.ones_like(weight)


@somefunc_backward.register_fake
def _(grad_output, input_, weight, shape):
    return torch.empty_like(weight)


def a_func(grad_output, input_, weight_, shape):
    return torch.ones_like(input_.sum() * weight_)


class SomeFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, normalized_shape):
        ctx.normalized_shape = normalized_shape
        input_ = input.contiguous()
        weight_ = weight.contiguous()
        output = somefunc_forward(input_, weight_, ctx.normalized_shape)
        ctx.save_for_backward(input_, weight_)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight_ = ctx.saved_tensors
        # grad_weight = a_func(grad_output, input_, weight_, ctx.normalized_shape)
        grad_weight = somefunc_backward(
            grad_output.contiguous(),
            input_,
            weight_,
            ctx.normalized_shape,
        )
        return None, grad_weight, None


class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(7))

    def forward(self, x):
        return SomeFunc.apply(x, self.weight, [7])


model = MyModel()
torch._logging.set_logs(dynamo=logging.DEBUG, aot=logging.DEBUG, graph_code=True)


def aot_print_backend(gm, sample_inputs):
    # Forward compiler capture
    def fw(gm, sample_inputs):
        print(f"----- fw")
        gm.print_readable()
        return make_boxed_func(gm.forward)

    # Backward compiler capture
    def bw(gm, sample_inputs):
        print(f"----- bw")
        gm.print_readable()
        return make_boxed_func(gm.forward)

    # Call AOTAutograd
    gm_forward = aot_module_simplified(
        gm, sample_inputs, fw_compiler=fw, bw_compiler=bw
    )
    return gm_forward


model = torch.compile(
    model,
    backend=aot_print_backend,
    dynamic=False,
)
out = model(torch.rand((128, 4, 7)))
out.mean().backward()