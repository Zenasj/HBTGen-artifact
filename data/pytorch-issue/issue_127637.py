import torch.nn as nn

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
torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.capture_scalar_outputs = True
torch._inductor.config.group_fusion = True







from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()



    def forward(self, memory_mask):
        x = torch._C._nn.pad(memory_mask, (7, 7, 7, 7), 'constant', 0);  memory_mask = None
        return (x,)


mod = Repro()

def load_args(reader):
    buf0 = reader.storage('9c3d20b7479d25692d89ff2d20ec7c5feb392eb5', 18496, device=device(type='cuda', index=0))
    reader.tensor(buf0, (1, 1, 68, 68), is_leaf=True)  # memory_mask
load_args._version = 0

if __name__ == '__main__':
    from torch._dynamo.repro.after_dynamo import run_repro
    run_repro(mod, load_args, accuracy=False, command='run',
        save_dir='/workspace/checkpoints', autocast=True, backend='inductor')