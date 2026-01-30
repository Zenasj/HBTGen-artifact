import torch.nn as nn

import torch
import torch.distributed as dist
import torch.nn.functional as F
from functorch import make_fx
import os

import torch.distributed._functional_collectives as ft_c
from torch.testing._internal.common_distributed import (
    spawn_threads_and_init_comms,
)
from torch._inductor.compile_fx import compile_fx_inner

def my_fun(a, b):
    c = a * 3
    tensors = ft_c.all_reduce_coalesced([a, c, b], "sum", [0])
    return ((tensors[1] + tensors[0] + tensors[2]).sum(), )

@spawn_threads_and_init_comms(world_size=1)
def inductor_main(self):

    x = torch.arange(4).cuda() * (dist.get_rank() + 1)
    y = torch.arange(4).cuda() * (dist.get_rank() + 1)
    x = x.to(torch.float)
    y = y.to(torch.float) * 0.5
    res = make_fx(my_fun)(x, y)
    print(f"fx graph:\n{res.graph}")
    ind = compile_fx_inner(res, [x, y])
    print(f"inductor done:\n{ind}")

os.environ["PROXY_TENSOR_TRACING"] = "1"
os.environ["TORCH_COMPILE_DEBUG"] = "1"
torch._dynamo.config.output_code = True

if __name__ == "__main__":
    inductor_main(None)