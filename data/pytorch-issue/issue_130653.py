import torch
from torch import Tensor
from torch.library import custom_op
import numpy as np


def to_numpy(tensor):
    return tensor.cpu().numpy()

class NumpySort(torch.autograd.Function):
    @staticmethod
    def forward(x, dim, /, k=1):
        device = x.device
        x = to_numpy(x)
        ind = np.argsort(x, axis=dim)
        ind_inv = np.argsort(ind, axis=dim)
        result = np.take_along_axis(x, ind, axis=dim)
        return (
            torch.tensor(result, device=device),
            torch.tensor(ind, device=device),
            torch.tensor(ind_inv, device=device),
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def vmap(info, in_dims, x, dim, /, k):
        x_bdim, _ = in_dims

        x = x.movedim(x_bdim, 0)
        dim = dim if dim >= 0 else dim + x.dim() - 1
        result = NumpySort.apply(x, dim + 1)

        return NumpySort.apply(x, dim + 1), (0, 0, 0)

def numpy_sort(x, dim=-1):
    result, _, _ = NumpySort.apply(x, dim, k=3)
    return result

x = torch.randn(2, 3)
result = torch.vmap(numpy_sort)(x)