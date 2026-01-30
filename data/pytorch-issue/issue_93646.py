import torch.nn as nn

import torch
from torch import tensor, device
import torch.fx as fx
from torchdynamo.testing import rand_strided
from math import inf
from torch.fx.experimental.proxy_tensor import make_fx

class Repro(torch.nn.Module):
    def forward(self, arg24_1, arg183_1, where_self_11):
        convolution_backward_default_7 = torch.ops.aten.convolution_backward.default(where_self_11, arg183_1, arg24_1, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  where_self_11 = arg183_1 = arg24_1 = None
        getitem_22 = convolution_backward_default_7[1];  convolution_backward_default_7 = None
        return [getitem_22]
    
args = [((32, 128, 1, 1), (128, 1, 1, 1), torch.float32, 'cuda'), ((1, 128, 1, 1), (128, 1, 128, 128), torch.float32, 'cuda'), ((1, 32, 1, 1), (32, 1, 32, 32), torch.float32, 'cuda')]
args = [rand_strided(shape, stride, dtype, device) for shape, stride, dtype, device in args]
mod = make_fx(Repro())(*args)

from torchinductor.compile_fx import compile_fx_inner

compiled = compile_fx_inner(mod, args)
compiled(*args)