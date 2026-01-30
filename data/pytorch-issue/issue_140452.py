import torch.nn as nn

import sys
from functools import partial
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
torch._dynamo.config.optimize_ddp = False

from typing import Dict, Optional
import torch
from torch.nn import *


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.weight_normed_linear = torch.nn.utils.parametrizations.weight_norm(torch.nn.Linear(in_features=in_features, out_features=2)).cuda()
        self.linear = torch.nn.Linear(in_features=2, out_features=1).cuda()

    def forward(self, x_0):
        x_1 = self.weight_normed_linear(x_0)
        x_2 = self.linear(x_1)
        return (x_2,)

def load_args(in_features, reader):  
    buf0 = reader.storage('fbae9e314f27f66ab2f21026f411a176d6711e51', 9043968, device=device(type='cuda', index=0), dtype_hint=torch.float16)
    reader.tensor(buf0, (2, in_features), dtype=torch.float16, requires_grad=True) 

if __name__ == '__main__':

    for in_features in [1024, 1025]:
      torch.compiler.reset()

      mod = Repro(in_features)
      load_args_partial = partial(load_args, in_features)
      load_args_partial.version = 0

      from torch._dynamo.repro.after_dynamo import run_repro
      run_repro(mod,load_args_partial, accuracy=True, command='run',
              save_dir='/stuff/felixb/kernel_workspace/checkpoints', autocast=True, backend='inductor')