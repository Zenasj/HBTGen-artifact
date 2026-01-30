import torch
import torch.nn as nn
import torch.nn.functional as F

class SWSConv2d(nn.Module):
    r"""
    2D Conv layer with Scaled Weight Standardization.

    Characterizing signal propagation to close the performance gap in unnormalized ResNets
    https://arxiv.org/abs/2101.08692
    """
    def __init__(self,
                 in_channels: int, out_channels: int, kernel_size: int,
                 stride: int=1, padding: int=0, padding_mode: str='zeros', dilation=1, groups: int=1,
                 bias: bool=True):

        super().__init__()

        self._stride = stride
        self._padding = padding
        self._padding_mode = padding_mode
        self._dilation = dilation
        self._groups = groups

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.gain = nn.Parameter(torch.ones(out_channels, 1, 1, 1))

        # Init weights
        k = groups / (in_channels * kernel_size * kernel_size)
        torch.nn.init.uniform_(self.weight, -k, k)
        if bias:
            torch.nn.init.uniform_(self.bias, -k, k)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return TF.conv2d(input,
                         weight=self.weight.to(input.device, copy=True) * self.gain, bias=self.bias,
                         stride=self._stride,
                         padding=self._padding,
                         dilation=self._dilation,
                         groups=self._groups)

# Avoid to recompute parametrization for weights shared
with torch.nn.utils.parametrize.cached():
  pred = self.model.forward(img0, img1, solver_num_iters=solver_num_iters)