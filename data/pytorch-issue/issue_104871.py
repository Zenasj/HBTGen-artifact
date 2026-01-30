import torch
from typing import List

def fn(input):
    return torch.empty_like(input) # torch.empty(10)

def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    gm.graph.print_tabular()
    return gm.forward

x = torch.rand([10], dtype=torch.float)

compiled = torch.compile(fn, backend=my_compiler)
ret_compiled = compiled(x)

import torch
import torch._dynamo
from torch._functorch.aot_autograd import aot_module_simplified


def fn(input):
    # return torch.empty(10)
    return torch.empty_like(input)
                              

def toy_backend(gm, sample_inputs):
    def my_compiler(gm, sample_inputs):
        # <implement your compiler here>
        print("AOTAutograd produced a fx Graph in Aten IR:")
        gm.print_readable()
        return gm.forward

    # Invoke AOTAutograd
    return aot_module_simplified(gm, sample_inputs, fw_compiler=my_compiler)


x = torch.rand([10], dtype=torch.float)

compiled = torch.compile(fn, backend=toy_backend)
ret_compiled = compiled(x)

import torch
import torch._dynamo
from torch._functorch.aot_autograd import aot_module_simplified
from torch._decomp import core_aten_decompositions

# Backends can further finetune the decompositions if needed
# Available decompositions can be found in
# torch/_decomp/decompositions.py and torch/_refs/__init__.py
decompositions = core_aten_decompositions()
decompositions.update(
    torch._decomp.get_decompositions([
        # torch.ops.aten.addmm,
    ])
)

def toy_backend(gm, sample_inputs):
    def my_compiler(gm, sample_inputs):
        # <implement your compiler here>
        print("AOTAutograd produced a fx Graph in Aten IR:")
        gm.print_readable()
        return gm.forward

    # Invoke AOTAutograd
    return aot_module_simplified(gm, sample_inputs, decompositions=decompositions, fw_compiler=my_compiler)

def fn(input):
    return torch.empty_like(input)


x = torch.rand([10], dtype=torch.float)

compiled = torch.compile(fn, backend=toy_backend)
ret_compiled = compiled(x)