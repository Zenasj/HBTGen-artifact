import torch
from torch._functorch.aot_autograd import aot_module_simplified
from torch._decomp import core_aten_decompositions

decompositions = core_aten_decompositions()

def toy_backend(gm, sample_inputs):
    def my_compiler(gm, sample_inputs):
        gm.print_readable()
        return gm

    # Invoke AOTAutograd
    return aot_module_simplified(
        gm,
        sample_inputs,
        decompositions=decompositions,
        fw_compiler=my_compiler
    )

def run(input):
  return torch.bernoulli(input, 0.5)

input = torch.randn(8, 32)

out = run(input)
print("EAGER OK")

fn = torch.compile(backend=toy_backend)(run)
out = fn(input)