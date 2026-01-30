import torch.nn as nn

import torch.nn.functional as F
import torch
import torch._dynamo as dynamo
from torch._functorch.aot_autograd import aot_module_simplified

dynamo.config.specialize_int = False
dynamo.reset()

def my_aot_compiler(gm, example_inputs):
    def my_compiler(gm, example_inputs):
        print(gm.code)
        return gm.forward

    # Invoke AOTAutograd
    return aot_module_simplified(
        gm,
        example_inputs,
        fw_compiler=my_compiler
    )

def my_example(t, p):
  m = torch.nn.ConstantPad2d(p, 3.5)
  out = m(t)
  return out   

compiled_fn = torch.compile(backend=my_aot_compiler, dynamic=True)(my_example)

t = torch.empty(2, 3, 4).requires_grad_(True)
p = (1, 1, 0, 0)
r = compiled_fn(t, p)
print(f"Output (1st) size = {r.size()}")

t2 = torch.empty(2, 3, 4).requires_grad_(True)
p2 = (2, 2, 0, 0)
r2 = compiled_fn(t2, p2)
print(f"Output (2nd) size = {r2.size()}")

t3 = torch.empty(2, 3, 4).requires_grad_(True)
p3 = (3, 3, 0, 0)
r3 = compiled_fn(t3, p3)
print(f"Output (3rd) size = {r3.size()}")

specialize_int_list