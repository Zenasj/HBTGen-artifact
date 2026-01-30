import torch
from torch.export import export
from torch.export._trace import _export
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.
import copy
import io
import random
import unittest
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from float8_experimental.float8_dynamic_linear import Float8DynamicLinear
from float8_experimental.float8_linear_utils import swap_linear_with_float8_linear
from float8_experimental.float8_tensor import Float8Tensor
from float8_experimental.float8_utils import compute_error
random.seed(0)
torch.manual_seed(0)
is_H100 = torch.cuda.is_available() and torch.cuda.get_device_capability() >= (9, 0)
import torch.nn.utils.parametrize as parametrize
# NOTE: we should upstream this directly into export and make it more automatic!
class UnwrapTensorSubclass(torch.nn.Module):
    def forward(self, *tensors):
        todo = list(tensors)
        for tp, meta, inner_tensors in reversed(self.rebuild_stack):
            nb_tensor = len(inner_tensors)
            inner_tensors = {a: b for a, b in zip(inner_tensors, todo[-nb_tensor:])}
            todo = todo[nb_tensor:]
            rebuilt = tp.__tensor_unflatten__(inner_tensors, meta, None, None)
            todo.append(rebuilt)
        assert len(todo) == 1
        return todo[0]
    def right_inverse(self, tensor):
        assert type(tensor) is not torch.Tensor
        rebuild_stack = []
        plain_tensors = []
        todo = [tensor]
        while todo:
            obj = todo.pop()
            inner_tensors, metadata = obj.__tensor_flatten__()
            rebuild_stack.append((type(obj), metadata, inner_tensors))
            for attr_name in inner_tensors:
                val = getattr(obj, attr_name)
                if type(val) is torch.Tensor:
                    plain_tensors.append(val)
                else:
                    assert isinstance(val, torch.Tensor)
                    todo.append(val)
        self.rebuild_stack = rebuild_stack
        return plain_tensors
def unwrap_tensor_subclass(model, filter_fn=None):
    for name, child in model.named_children():
        if (
            isinstance(child, Float8DynamicLinear) and
            hasattr(child, "weight") and
            type(child.weight) is not torch.Tensor and
            isinstance(child.weight, torch.Tensor)
        ):
            parametrize.register_parametrization(child, "weight", UnwrapTensorSubclass())
        unwrap_tensor_subclass(child)
    return model
class FeedForward(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w1 = nn.Linear(4096, 14336, bias=False)
        self.w3 = nn.Linear(4096, 14336, bias=False)
        self.w2 = nn.Linear(14336, 4096, bias=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()
export_model = FeedForward().to("cuda")
swap_linear_with_float8_linear(
    export_model,
    Float8DynamicLinear,
    from_float_kwargs={"pre_quantize_weight": True},
)
export_model = unwrap_tensor_subclass(export_model)
batch_size = 4
num_tokens = 1024
embedding_dim = 4096
input_tensor = torch.randn(
    batch_size, num_tokens, embedding_dim, device="cuda", dtype=torch.float32
)
example_args = (input_tensor,)
# NOTE: this breaks unless we use strict=False, pre_dispatch=False!
exported_program: torch.export.ExportedProgram = _export(
    export_model,
    example_args,
    strict=False,
    pre_dispatch=False,
)
with torch.no_grad():
    so_path = torch._inductor.aot_compile(exported_program.module(), example_args)
    print(so_path)