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

def my_example(t1, t2, d):
  out = torch.add(t1, t2, alpha=d)
  return out

compiled_fn = torch.compile(backend=my_aot_compiler, dynamic=True)(my_example)

t1 = torch.arange(3, dtype=torch.float32).requires_grad_(True)
t2 = torch.arange(3, dtype=torch.float32).requires_grad_(True)

ra = compiled_fn(t1, t2, 5)
print(f"ra = {ra}")

ra = compiled_fn(t1, t2, 6)
print(f"ra = {ra}")