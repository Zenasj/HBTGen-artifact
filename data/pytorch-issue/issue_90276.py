import torch
import torch._dynamo as dynamo
from torch._dynamo.backends.common import aot_autograd

from typing import List

def my_backend(gm: torch.fx.GraphModule,
                    example_inputs: List[torch.Tensor]):
    gm.print_readable()
    return gm

my_backend = aot_autograd(fw_compiler=my_backend) 

@dynamo.optimize(backend=my_backend)
def f(x):
    return x * x

print(f(torch.rand(3)))