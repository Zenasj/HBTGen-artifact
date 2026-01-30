import torch.nn as nn

import torch

@torch.compile(backend='relu_compile_error_TESTING_ONLY')
def fn(x):
    return torch.relu(x)

fn(torch.randn(5, 5))

from math import inf
import torch
from torch import tensor, device
import torch.fx as fx
import torch._dynamo
from torch._dynamo.testing import rand_strided
from torch._dynamo.debug_utils import run_fwd_maybe_bwd

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config









from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()



    def forward(self, L_L_x_ : torch.Tensor):
        l_x_ = L_L_x_
        relu = torch.relu(l_x_);  l_x_ = None
        return (relu,)


mod = Repro()

def load_args(reader):
    buf0 = reader.storage('3923b434f408c11083be64af0e6c46fa354c898e', 100)
    reader.tensor(buf0, (5, 5), is_leaf=True)  # L_L_x_
load_args._version = 0

if __name__ == '__main__':
    from torch._dynamo.repro.after_dynamo import run_repro
    run_repro(mod, load_args, accuracy=False, command='run',
        save_dir='/data/users/williamwen/pytorch/checkpoints', autocast=False, backend='relu_compile_error_TESTING_ONLY')