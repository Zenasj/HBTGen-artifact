import torch
import torch._dynamo as dynamo
from torch._functorch.aot_autograd import aot_module_simplified
from torch._decomp import core_aten_decompositions

def toy_backend(gm, sample_inputs):
    def my_compiler(gm, sample_inputs):
        print(gm.code)
        return gm.forward

    # Invoke AOTAutograd
    return aot_module_simplified(
        gm,
        sample_inputs,
        decompositions=core_aten_decompositions(),
        fw_compiler=my_compiler
    )

def toy_example():
    t1 = torch.rand((8, 8))
    t2 = torch.empty(8, 8)
    return t1 + t2

compiled_fn = torch.compile(backend=toy_backend)(toy_example)
r = compiled_fn()