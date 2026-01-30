import torch.nn as nn

from typing import Tuple, Literal

import math
import torch
from torch import Tensor


def symmetric_pad(tensor: Tensor, pad: Tuple[int, int], axis: int):
    prev = torch.arange(+(pad[0] - 1), -1, -1, device=tensor.device)
    last = torch.arange(-1, -(pad[1] + 1), -1, device=tensor.device)
    curr = torch.arange(00, tensor.shape[axis], +1, device=tensor.device)
    return torch.index_select(tensor, dim=axis, index=torch.concat([prev, curr, last], dim=0))


def extension_pad(tensor: Tensor, pad: Tuple[int, int], axis: int, mode: Literal['constant', 'reflect', 'symmetric', 'replicate', 'circular'] = 'symmetric', cval: float = 0.0):
    if mode == 'symmetric':
        return symmetric_pad(tensor, pad, axis)
    else:
        return torch.nn.functional.pad(tensor.movedim(axis, -1), pad, mode=mode, value=cval).movedim(axis, -1)


def convolve1d(tensor: Tensor, weight: Tensor, axis: int, mode: Literal['constant', 'reflect', 'symmetric', 'replicate', 'circular'] = 'symmetric', cval: float = 0.0):
    weight = weight.flatten().to(device=tensor.device)
    weight = weight.view(+1, +1, -1)

    pad = (weight.numel() - 1) / 2
    pad = (math.floor(pad), math.ceil(pad))

    tensor = tensor.movedim(axis, -1).unsqueeze(-2)
    # Record Shape
    shape = tensor.shape
    if tensor.ndim > 3:
        tensor = tensor.flatten(0, -3)
    elif tensor.ndim < 2:
        tensor = tensor.unsqueeze(0)
    tensor = extension_pad(tensor, pad, axis=-1, mode=mode, cval=cval)

    result = torch.nn.functional.conv1d(tensor, weight)
    result = result.squeeze()
    result = result.unflatten(0, shape[:-2]) if len(shape) > 3 else result
    return result.movedim(-1, axis)


if __name__ == '__main__':
    with torch.device(0):
        a = torch.randn([10, 20, 30])
        p = convolve1d(a, torch.randn(10), -2, mode='symmetric')
        print(p.shape)