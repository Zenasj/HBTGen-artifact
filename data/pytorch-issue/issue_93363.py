import numpy as np

import torch._dynamo as dynamo
import torch
from torch._functorch.aot_autograd import aot_module_simplified

dynamo.reset()

def toy_backend(gm, sample_inputs):
    def my_compiler(gm, sample_inputs):
        print(gm.code)
        return gm.forward

    # Invoke AOTAutograd
    return aot_module_simplified(
        gm,
        sample_inputs,
        fw_compiler=my_compiler
    )

def foo(a):
    a.add_(1.0)
    return

compiled_fn = torch.compile(backend=toy_backend)(foo)

# Invoke the torch.compile version
a1 = torch.ones([4], requires_grad = True).add(0.)
a1_view = a1[0::2]
assert(a1_view.requires_grad)
out1 = compiled_fn(a1_view)
print(f"a1 after call to compiled_fn = {a1}")

# Invoke without the torch.compile version
a2 = torch.ones([4], requires_grad = True).add(0.)
a2_view = a2[0::2]
out2 = foo(a2_view)
print(f"a2 after call to foo = {a2}")

print("Results match? ", torch.allclose(a1, a2, atol = 0.001, rtol = 0.001))

inp = torch.ones(2)
inp_view = inp.as_strided((2,), (2,))

def mutate_data(view):
    view.add_(1)

def mutate_data_and_metadata(view):
     view.add_(1)
     view.as_strided_((2,), (1,))
     
functionalize(mutate_data)(inp_view)
functionalize(mutates_data_and_metadata)(inp_view)