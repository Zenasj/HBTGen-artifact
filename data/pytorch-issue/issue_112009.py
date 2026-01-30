import torch.nn as nn

import torch
import os
from torch.fx.experimental.proxy_tensor import make_fx
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import torch.multiprocessing as mp
from torch._functorch.aot_autograd import aot_module_simplified
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

def toy_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print(gm.graph)
    return gm

def toy_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    return aot_module_simplified(gm, example_inputs,fw_compiler=toy_compiler)

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = funcol.all_gather_tensor(x, 0, group=[0,1],tag='test1')
        return x

def example(rank, world_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    mod = MyModule()
    opt_mod = torch.compile(mod, dynamic=True, fullgraph=True, backend=toy_backend)
    x = torch.randn(10, 100)
    out = opt_mod(x)
    print(out)


def main():
    world_size = 2
    mp.spawn(example,
             args=(world_size, ),
             nprocs=world_size,
             join=True)


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29516"
    main()