import torch
import os
import torch.nn as nn
import torch._dynamo
from typing import List
from torch._dynamo.backends.registry import register_backend
from torch._dynamo import compiled_autograd
from torch._dynamo.backends.common import aot_autograd
#import habana_frameworks.torch.core as htcore
#import habana_frameworks.torch.dynamo.compile_backend

def custom_inner_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("custom backend called")
    for input in example_inputs:
        print(input)
    #print(gm.graph)
    return gm.forward

@register_backend
def custom_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    return aot_autograd(
            fw_compiler=custom_inner_compiler,
            bw_compiler=custom_inner_compiler,
            )(gm, example_inputs)


def test_randperm(n):
    def fn(n, g):
        return torch.randperm(n, generator=g)
    seed = 1234
    torch.manual_seed(seed)
    g = None#torch.Generator()
    fn = torch.compile(fn, dynamic=True, backend="custom_backend")
    hpu_res1 = fn(n, g)

test_randperm(10)