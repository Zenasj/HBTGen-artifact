import torch.nn as nn

import sys
import torch
from torch import distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.experimental import context_parallel, _attention

# CLI args: [seqlen-to-headdim ratio, enable_load_balance, is_causal]
multiplier, balance, causal = int(sys.argv[1]), eval(sys.argv[2]), eval(sys.argv[3])
_attention._cp_options.enable_load_balance = balance

# init mesh && seed.
mesh = init_device_mesh('cuda', (2,), mesh_dim_names=['cp'])
torch.cuda.set_device(mesh.get_rank())
torch.manual_seed(0)

# try SDPA + backprop
with torch.device('cuda'):
    S,D = multiplier*64,64
    x,y = torch.randn(1,1,S,D), torch.randn(1,1,S,D)
    l = torch.nn.Linear(D, 3*D, bias=False)
    with context_parallel(mesh['cp'], buffers=[x,y], buffer_seq_dims=[2,2]):
        q,k,v = l(x).chunk(3, dim=-1)
        o = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)
        torch.nn.functional.mse_loss(o, y).backward()

dist.destroy_process_group(); exit()