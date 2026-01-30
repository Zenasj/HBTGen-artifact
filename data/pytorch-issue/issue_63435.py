import torch.nn as nn

import torch
import torch.nn.functional as F

class TestTensor(object):
    def __init__(self, weight):
        self.weight = weight

    def __torch_function__(self, func, _, args=(), kwargs=None):
        print(func)
        print(func == F.group_norm)

features = TestTensor(torch.randn(3,3))
F.group_norm(features, 3)

features = torch.randn(3,3)
weight = TestTensor(torch.randn(3))
F.group_norm(features, 3, weight=weight)

def group_norm(
    input: Tensor, num_groups: int, weight: Optional[Tensor] = None, bias: Optional[Tensor] = None, eps: float = 1e-5
) -> Tensor:
    r"""Applies Group Normalization for last certain number of dimensions.

    See :class:`~torch.nn.GroupNorm` for details.
    """
    if has_torch_function_unary(input):
        return handle_torch_function(group_norm, (input,), input, num_groups, weight=weight, bias=bias, eps=eps)