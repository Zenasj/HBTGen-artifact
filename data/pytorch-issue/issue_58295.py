from torch.testing._internal.common_methods_invocations import (SampleInput, make_tensor)
import torch
from functools import partial

make_arg = partial(make_tensor, device='cpu', dtype=torch.float, requires_grad=True)

print("Scalar Tensor", SampleInput(make_arg(())).summary(), "\n")

print("2-D Tensor", SampleInput(make_arg((2, 3))).summary(), "\n")

inp = [make_arg((2, 3)), make_arg((2, 3)), make_arg((2, 3))]

print("Tensor List", SampleInput(inp).summary(), "\n")

lst = [make_arg((3, 3)), make_arg((2, 3)), make_arg((1, 3))]

print("kwargs with tensor list", SampleInput(inp, kwargs={"other": make_arg((3, 3)), "t_list": lst}).summary(), "\n")

print("Args with 2 tensors and Python int", SampleInput(inp, args=(
    make_arg((3, 4)), make_arg((4, 3)), 0), name="test_sample").summary(), "\n")


print("Args tuple with tensor list and name", SampleInput(inp, args=(lst, 0), name="test_sample").summary(), "\n")

print("Args list  with tensor list, kwargs and name", SampleInput(inp, args=[lst, 0], kwargs={"other": make_arg(
    (3, 3)), "t_list": lst}, name="test_sample").summary(), "\n")